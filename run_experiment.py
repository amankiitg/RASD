"""
RASD Experiment Runner
======================
Reads configs/ablations.yml (or any YAML passed via --config), expands the
ablation grid, and runs each (level × seed) combination. Logs every run to
wandb and appends a row to results/ablations/ablations.csv.

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

    # Resume: skip runs already present in results/ablations/ablations.csv
    python run_experiment.py --config configs/ablations.yml --resume
"""

import argparse
import csv
import itertools
import json
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

RESULTS_DIR = Path("results/ablations")
RESULTS_CSV  = RESULTS_DIR / "ablations.csv"

CSV_FIELDS = [
    "run_id", "group", "level_id", "seed",
    "target_model_name", "draft_model_name",
    "spec_steps", "kv_block_size", "prefetch_depth",
    "context_length", "dtype",
    # metrics
    "tokens_generated", "time_sec", "throughput_tps",
    "acceptance_rate", "mean_latency_ms", "ttft_ms",
    "gpu_peak_mem_mb",
    "n_rounds", "status", "error",
]


def _per_token_sidecar_path(output_csv: str | Path, run_id: str) -> Path:
    """Where C13 per-position trace .jsonl lands for a given run.

    Sits next to the CSV under a `per_token/` subdir so the directory
    layout stays grep-friendly:

        results/ablations/ablations_r65.csv
        results/ablations/per_token/A2_k4_s42.jsonl
        results/ablations/per_token/A3_block2048_s123.jsonl
        ...
    """
    return Path(output_csv).resolve().parent / "per_token" / f"{run_id}.jsonl"


def write_per_token_sidecar(
    trace: list[dict] | None, output_csv: str | Path, run_id: str
) -> Path | None:
    """Write per-position trace records (C13) as one-record-per-line JSONL.

    Returns the path written, or None if `trace` is empty/None.
    """
    if not trace:
        return None
    path = _per_token_sidecar_path(output_csv, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in trace:
            f.write(json.dumps(record) + "\n")
    return path


def _profiler_sidecar_path(output_csv: str | Path, run_id: str) -> Path:
    """Where C7 profiler summary lands for a given run.

    Sits next to the CSV under a `profiler/` subdir, mirroring the
    `per_token/` layout for per-position traces:

        results/final/final_matrix.csv
        results/final/profiler/M4_ctx128k_s42.json
        results/final/profiler/M4_ctx256k_s42.json
        ...
    """
    return Path(output_csv).resolve().parent / "profiler" / f"{run_id}.json"


def write_profiler_sidecar(
    summary: dict | None, output_csv: str | Path, run_id: str
) -> Path | None:
    """Write a RoundProfiler.summary dict (C7) to <profiler>/<run_id>.json.

    Returns the path written, or None if `summary` is empty/None.
    """
    if not summary:
        return None
    path = _profiler_sidecar_path(output_csv, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return path


# ---------------------------------------------------------------------------
# Config loading & grid expansion
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_configs(cfg: dict, groups: Optional[List[str]], debug: bool,
                      seed_filter: Optional[List[int]] = None) -> List[dict]:
    """Expand the YAML into a flat list of run dicts (one per level × seed).

    Top-level YAML keys are either the canonical scaffolding keys
    (`defaults`, `canary`) OR an ablation group. M3 used `A1..A5`; M4
    introduces `SMOKE` (long-context smokes) and `M4` (final matrix).
    Filtering by an `A*` prefix would silently drop M4's groups. Instead
    we exclude the known scaffolding keys and treat the rest as groups.
    """
    NON_GROUP_KEYS = {"defaults", "canary"}

    defaults = deepcopy(cfg["defaults"])
    seeds    = defaults.pop("seeds")
    if seed_filter:
        seeds = [s for s in seeds if s in seed_filter]

    ablation_keys = [k for k in cfg if k not in NON_GROUP_KEYS]
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
    # Tick gate: wait for rank 0 to signal each round before submitting
    # P2P. Without this, peers race ahead of rank 0's slower generate()
    # loop (which does real LLM inference), causing NCCL sequence number
    # divergence and deadlock at block_size=2048.
    tick = torch.zeros(1, dtype=torch.int32, device=device)
    for _round in range(max_rounds):
        # Wait for rank 0's tick — keeps peer paced to rank 0's generate loop
        dist.recv(tick, src=0)
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

    Post-R3 architecture (2026-05-06): ALL ranks run the full
    RASDInference.generate() pipeline in lockstep. Ring attention happens
    inside LlamaAttention.forward; there is no master/slave pattern. Only
    rank 0 initialises wandb and writes the result CSV.

    The previous _ring_peer_loop pattern (rank 0 = master, ranks 1..N-1 =
    passive P2P participants) was for the old AsyncKVRingPrefetcher; that
    code was deleted in commit 09f7d98 when ring moved into the attention
    forward.
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
    else:
        torch.cuda.set_device(0)

    row = {f: run.get(f, "") for f in CSV_FIELDS}
    row["status"] = "error"
    row["error"]  = ""

    # Only rank 0 logs to wandb (avoids 8x duplicate runs in the project).
    wb_run = init_wandb(run, wandb_project) if local_rank == 0 else None
    try:
        cfg = RASDConfig(
            target_model_name = run["target_model_name"],
            draft_model_name  = run["draft_model_name"],
            target_revision   = run.get("target_revision") or None,
            draft_revision    = run.get("draft_revision") or None,
            spec_steps        = int(run["spec_steps"]),
            kv_block_size     = int(run["kv_block_size"]),
            prefetch_depth    = int(run["prefetch_depth"]),
            max_new_tokens    = int(run.get("max_new_tokens", 256)),
            dtype             = run.get("dtype", "bfloat16"),
            quantize_draft    = bool(run.get("quantize_draft", True)),
            quantize_target   = bool(run.get("quantize_target", False)),
            temperature       = float(run.get("temperature", 1.0)),
            top_p             = float(run.get("top_p", 1.0)),
            context_length    = int(run.get("context_length", 0)),
            # C2b RoPE strategy. Default "linear" preserves M3 behavior.
            # Use "yarn" for M4 1M context (factor=256 over Llama-2's 4k).
            rope_type         = str(run.get("rope_type", "linear")),
            # C11 NF4 KV-cache. Default False -> M3 byte-identical.
            # Phase C P3.5 final matrix uses kv_quant=true at all contexts.
            kv_quant          = bool(run.get("kv_quant", False)),
            seed              = int(run["seed"]),
            debug             = bool(run.get("debug", False)),
            # C13 per-position trace (M4 Phase A2). Default off so M3
            # replay is byte-identical when --log-per-token is unset.
            log_per_token     = bool(run.get("log_per_token", False)),
            # C6 generation checkpoint/resume (Phase C blocker #2 from
            # 2026-05-10 third-pass review). Default 0 -> disabled.
            # 1M cells set checkpoint_every>=1 to recover from crashes
            # — without this every crash on a 120-min cell loses the
            # full run. checkpoint_dir defaults to <output_csv>/../
            # checkpoints/ so per-run paths are predictable for resume.
            checkpoint_every  = int(run.get("checkpoint_every", 0)),
            checkpoint_dir    = (
                run.get("checkpoint_dir")
                or str(Path(output_csv).resolve().parent / "checkpoints")
            ),
            run_id            = run["run_id"],
        )
        engine = RASDInference(cfg)
        prompt = build_prompt(int(run.get("context_length", 65536)), engine.tokenizer)

        # M4 C7 — torch.profiler wrap (default off). Enabled per-row via
        # run["profile"] = True (set from --profile CLI flag below).
        profile_enabled = bool(run.get("profile", False)) and local_rank == 0
        if profile_enabled:
            from src.analysis.profiler import RoundProfiler
            with RoundProfiler(enabled=True) as _prof:
                _, metrics = engine.generate_text(prompt)
            metrics["_profiler_summary"] = _prof.summary
        else:
            _, metrics = engine.generate_text(prompt)

        # Pull the per-position trace out of the metrics dict before
        # wandb logging — it's a list-of-dicts, not a wandb-loggable
        # scalar. Only rank 0 receives a non-None trace (others get
        # None per RASDInference.generate's rank-0 guard).
        trace = metrics.pop("per_token_trace", None)
        prof_summary = metrics.pop("_profiler_summary", None)
        if local_rank == 0:
            sidecar = write_per_token_sidecar(trace, output_csv, run["run_id"])
            if sidecar is not None:
                log.info("Wrote per-position trace: %s (%d records)",
                         sidecar, len(trace))
            prof_path = write_profiler_sidecar(prof_summary, output_csv, run["run_id"])
            if prof_path is not None:
                log.info("Wrote profiler summary: %s "
                         "(compute=%.1fms, comm=%.1fms, idle=%.1fms)",
                         prof_path,
                         prof_summary["compute_us"] / 1000,
                         prof_summary["comm_us"]    / 1000,
                         prof_summary["idle_us"]    / 1000)

        row.update({
            "tokens_generated": metrics["tokens_generated"],
            "time_sec":         round(metrics["time_sec"], 4),
            "throughput_tps":   round(metrics["throughput_tps"], 2),
            "acceptance_rate":  round(metrics["acceptance_rate"], 4),
            "mean_latency_ms":  round(metrics["mean_latency_ms"], 3),
            "ttft_ms":          round(metrics["ttft_ms"], 3),
            "gpu_peak_mem_mb":  round(metrics["gpu_peak_mem_mb"], 1),
            "n_rounds":         metrics["n_rounds"],
            "status":           "ok",
        })
        if wb_run is not None:
            log_wandb(wb_run, metrics)
    except Exception as exc:
        row["error"] = str(exc)
        if wb_run is not None:
            import wandb; wb_run.finish(exit_code=1)

    # Write results BEFORE destroy_process_group — destroy can hang if peers
    # are slow to reach the same barrier, and we don't want to lose metrics.
    if local_rank == 0:
        # mkdir -p the output_csv's parent so writing the .tmp file
        # doesn't FileNotFoundError on first-run output dirs that
        # haven't been created yet (results/m4_smoke/, results/final/,
        # etc.). Bug discovered 2026-05-10 when the M4 smoke canary
        # generated successfully but failed to persist its result.
        Path(output_csv).resolve().parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv + f".{run['run_id']}.tmp", "w") as f:
            json.dump(row, f)

    if world_size > 1:
        dist.destroy_process_group()


def execute_run(run: dict, wandb_project: str, output_csv: str,
                nproc: int = 1, timeout_s: int = 3600) -> dict:
    """Run a single ablation in an isolated subprocess to keep CUDA context clean.

    GPU state is contaminated after a CUDA error — running each job in its own
    process guarantees a fresh device context regardless of previous failures.

    When nproc > 1, launches via torchrun so the ring attention distributed
    primitives (A3 kv_block_size, A4 prefetch_depth) actually exercise multi-GPU
    communication. With nproc=1 the ring is a no-op (world_size=1).

    `timeout_s`: hard wall-clock limit per run. Default 3600s (1 hr) is
    safe for ablation rows at ctx ≤ 64k. Phase C 1M cells need at least
    14400s (4 hr) — the smoke YAML's own comment estimates 120 min per
    1M cell. Raise via --timeout-per-run-s when launching long-context.
    """
    import json, subprocess, sys

    log.info("▶  %s  (seed=%d, nproc=%d, timeout=%ds)",
             run["run_id"], run["seed"], nproc, timeout_s)

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
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.error("✗  %s  TIMEOUT after %ds", run["run_id"], timeout_s)
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
    parser.add_argument("--seeds",    nargs="+", type=int,
                        help="Subset of seeds to run (e.g. 42). Default: all seeds in config.")
    parser.add_argument("--dry-run",  action="store_true", help="Print jobs without executing")
    parser.add_argument("--debug",    action="store_true", help="Enable RASD debug mode")
    parser.add_argument("--resume",   action="store_true", help="Skip runs already in results CSV")
    parser.add_argument("--wandb-project", default="rasd-ablations", help="wandb project name")
    parser.add_argument("--output",   default=str(RESULTS_CSV), help="Output CSV path")
    parser.add_argument("--nproc",    type=int, default=8,
                        help="GPUs per run (torchrun nproc_per_node). Use 1 for single-GPU. "
                             "Default 8 — required for ring attention A3/A4 ablations.")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save a generation checkpoint every N spec rounds. "
                             "0 = disabled (M3 byte-identical default). "
                             "Phase C 1M cells should set >= 4 — every crash "
                             "on a 120-min run otherwise loses the full run. "
                             "Path: <output_csv>/../checkpoints/<run_id>/")
    parser.add_argument("--abort-on-failure", action="store_true",
                        help="Stop the grid loop on the first row where "
                             "status != 'ok'. Default off — useful for the "
                             "M3 ablation matrix where OOM in one cell "
                             "shouldn't stop the rest. Phase C smoke stages "
                             "should set this so a 128k OOM doesn't burn "
                             "pod-$ on 512k/1M cells that are guaranteed "
                             "to OOM too. Resume from --resume picks up "
                             "where the abort happened.")
    parser.add_argument("--timeout-per-run-s", type=int, default=3600,
                        help="Hard wall-clock timeout per run in seconds. "
                             "Default 3600 (1 hr) is safe for ctx ≤ 64k. "
                             "Phase C 1M cells need >= 14400 (4 hr) — see "
                             "configs/m4_phase_c_long_smoke.yml comment "
                             "(~120 min per 1M cell). Killing a 1M run mid-way "
                             "wastes the entire pod-hour spent on it.")
    parser.add_argument("--log-per-token", action="store_true",
                        help="Enable C13 per-position acceptance trace sidecar "
                             "(.jsonl per run, source for Figure 4). "
                             "Default off so M3 replay stays byte-identical.")
    parser.add_argument("--profile", action="store_true",
                        help="Wrap each generate() call in a torch.profiler "
                             "context (C7 RoundProfiler). Writes per-run "
                             "compute/comm/idle JSON sidecar to "
                             "<output_dir>/profiler/<run_id>.json. Source for "
                             "Figure 3 (mentor stacked time breakdown). "
                             "Adds ~10%% overhead — run on a subset, not the "
                             "headline matrix.")
    # Internal: subprocess worker mode
    parser.add_argument("--_worker",  default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ---- Subprocess worker path ----
    if args._worker:
        run = json.loads(args._worker)
        output_csv = args.output
        _run_single_worker(run, args.wandb_project, output_csv)
        return

    cfg        = load_config(args.config)
    all_runs   = build_run_configs(cfg, args.groups, args.debug, seed_filter=args.seeds)
    # Propagate the --log-per-token flag onto every run dict so the
    # subprocess worker picks it up (each run is serialized via --_worker).
    if args.log_per_token:
        for r in all_runs:
            r["log_per_token"] = True
    if args.profile:
        for r in all_runs:
            r["profile"] = True
    if args.checkpoint_every > 0:
        for r in all_runs:
            # Per-run override wins; CLI flag is a default for runs that
            # don't specify their own checkpoint_every in the YAML
            r.setdefault("checkpoint_every", args.checkpoint_every)
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
            canary_row = execute_run(canary_run, args.wandb_project, str(output_csv),
                                     nproc=args.nproc, timeout_s=args.timeout_per_run_s)
            append_csv(output_csv, canary_row)
            _wait_gpu_idle()
            if canary_row.get("status") != "ok":
                log.error("CANARY FAILED — aborting ablation grid. Error: %s",
                          canary_row.get("error", "unknown"))
                log.error("Fix the issue and re-run. The canary will be skipped once it passes.")
                # Use sys.exit(1) instead of `return` so the calling
                # script (e.g., phase_c_pod_session.sh) sees a non-zero
                # exit and stops. Bug discovered 2026-05-10: a `return`
                # exits Python with code 0, so the master script
                # marked the stage as DONE and proceeded to the next
                # stage despite a broken canary.
                sys.exit(1)
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
        row = execute_run(run, args.wandb_project, str(output_csv),
                          nproc=args.nproc, timeout_s=args.timeout_per_run_s)
        append_csv(output_csv, row)
        # Wait for all GPU processes from this run to fully exit before starting
        # the next one — prevents CUDA/NCCL state pollution across runs.
        _wait_gpu_idle()
        # Phase C smoke stages: stop-hard-on-failure so a 128k OOM
        # doesn't burn pod-$ on 512k/1M cells we know will also OOM.
        # M3 ablation default is fail-tolerant (no abort) so a single
        # OOM cell doesn't drop the rest of the matrix.
        if args.abort_on_failure and row.get("status") != "ok":
            log.error(
                "✗  ABORT-ON-FAILURE: row %s status=%s; stopping grid "
                "(re-run with --resume to pick up after fixing).",
                run["run_id"], row.get("status"),
            )
            break

    # Summary
    import csv as _csv
    with open(output_csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    n_ok  = sum(1 for r in rows if r["status"] == "ok")
    n_err = sum(1 for r in rows if r["status"] != "ok")
    log.info("Done. %d succeeded, %d failed. Results: %s", n_ok, n_err, output_csv)


if __name__ == "__main__":
    main()

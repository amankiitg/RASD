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
import hashlib
import itertools
import json
import logging
import os
import random
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

def _guard_output_collision(output_csv: str | Path, stage_id: str,
                            overwrite: bool = False) -> None:
    """Refuse to write a stage's results over an artifact it does not own.

    Learned the hard way: the ARM4 driver wrote a stage named
    `pg19_short_target` with `--output results/mlsys/pg19_multiseed.csv`, which
    is the filename of an unrelated multi-seed AGGREGATE produced by the
    analysis pipeline. The stage would have overwritten it. `gpu_hours.csv`
    had the same problem from the other direction (a cross-run cumulative
    ledger vs a per-stage output).

    The contract: a per-stage result file must be named for exactly one stage.
    If the target already exists and its recorded stage (= the seed-0 canary's
    level_id / run_id prefix) does not match this stage, abort rather than
    silently clobber. Pass --overwrite-stage to deliberately re-run and
    replace the same stage.

    This is deliberately a hard failure: a silently overwritten aggregate is
    indistinguishable from a correct result afterwards, which is the failure
    class this project keeps having to reconstruct from git history.
    """
    path = Path(output_csv)
    if not path.exists() or overwrite:
        return
    try:
        import pandas as pd
        df = pd.read_csv(path)
        existing = sorted({str(x) for x in df.get("level_id", []) if str(x)})
    except Exception:
        existing = []
    # A stage owns the file if at least one of its rows was produced by a
    # level under this stage's id, or the file has no level_id column at all
    # (then it is not a per-stage result CSV and is not ours to check).
    if not existing:
        return
    if any(stage_id.split("_")[0] in lv or lv.startswith(stage_id.split("_")[0])
           for lv in existing):
        return
    raise SystemExit(
        f"REFUSING to write {path} for stage {stage_id!r}: the file already "
        f"holds rows from {existing[:4]}{'...' if len(existing) > 4 else ''}. "
        f"Give this stage its own filename (recommended) or pass "
        f"--overwrite-stage if you really mean to replace it."
    )


CSV_FIELDS = [    "run_id", "group", "level_id", "seed",
    "target_model_name", "draft_model_name",
    "spec_steps", "kv_block_size", "prefetch_depth",
    "context_length", "dtype",
    # metrics
    "tokens_generated", "time_sec", "throughput_tps",
    "acceptance_rate", "mean_latency_ms", "ttft_ms",
    "gpu_peak_mem_mb",
    "n_rounds", "status", "error",
    # MLSys Phase 1/3 — draft isolation provenance. Appended (not
    # interleaved) so existing CSV headers keep their column order and
    # --resume against older result files still aligns.
    "draft_window_cap", "draft_dtype",
    # MLSys Phase A — generation-length provenance. ARM4 uses 128 (2x the
    # M3/M4 value of 64) so the per-round dip test has more rounds; recording
    # it makes that visible in the results rather than inferred. Appended for
    # the same --resume alignment reason as above.
    "max_new_tokens",
    # MLSys A6 — per-row provenance. Provenance is only meaningful if it
    # travels with the row: a revision pin that lives only in a config file
    # cannot be checked against an old CSV later.
    "target_revision", "draft_revision",
    "prompt_tokens", "prompt_sha256",
    # MLSys analysis-plan fields. Appended for the same --resume alignment
    # reason as above. `doc_id` names the PG-19 book a row belongs to, which
    # the plan makes the unit of independence; the decoding contract is
    # recorded because losslessness is only defined under greedy +
    # ignore_eos, and a row from a different contract must not be compared
    # against one from this one.
    "prompt_source", "doc_id", "temperature", "top_p", "ignore_eos",
    # Pairing key. Spec, target-full and target-short rows of the same rung and
    # document carry the SAME pair_id, so the paired bootstrap can select the
    # intended target arm by id instead of by position. Position-based
    # selection (`first row with spec_steps == 0`) silently averaged or
    # discarded the other arm whenever a document had more than one partner,
    # and nothing in the output said which partner had been used.
    "pair_id", "arm_role",
    # Which arm of a multi-arm intervention this row is, when a stage declares
    # one. `arm_role` says spec/target, which is not enough to tell the arms of
    # rope_intervention_128k apart: all three are speculative, and the stage's
    # whole purpose is to compare them.
    "rope_arm",
    "generated_tokens_sha256",
    # Target quality beside acceptance (plan 4.3). Blank when not measured,
    # which is distinguishable from a measured zero.
    "target_ppl", "ppl_tokens",
    # Sequence the engine actually built (prompt + leading BOS + generated).
    # Recorded so the plan's identity can be checked from the CSV instead of
    # trusted, since the plan and the manifest previously disagreed about
    # whether the BOS was inside `context_length`.
    "sequence_tokens",
    # Pre-registered decode rate: generated tokens over the post-prefill wall.
    # The same definition is used in both arms, so any convention cancels in
    # the paired ratio. Preferred over the end-to-end rate for the primary
    # metric because prefill is identical across arms and under 5% of the
    # decode wall (measured 0.22% at 128k, 0.54% at 512k).
    "decode_tps",
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


# How far the source length may be adjusted to make the engine's prompt exactly
# the rung length. Measured drift on the staged pool is a constant 7 tokens for
# the one book that drifts; the bound only has to exceed it comfortably.
MAX_PROMPT_SHIFT = 4096


def _generated_tokens_dir(output_csv: str | Path) -> Path:
    return Path(output_csv).resolve().parent / "tokens"


def write_generated_tokens_sidecar(
    output_csv: str | Path, run_id: str, token_ids: list[int] | None,
    provenance: dict | None = None,
) -> Path | None:
    """Write the raw generated token IDs for a run.

    Losslessness is a token-level claim, so the token IDs must survive the run.
    Only decoded text was saved before, and text is not injective: a decode then
    re-encode need not round-trip, so a text comparison could neither prove nor
    disprove token-level agreement.

    One JSON file per run, including the provenance needed to pair it with its
    counterpart (prompt hash, context, generation length, decoding contract).
    The pairing fields are duplicated here rather than looked up from the CSV so
    that a sidecar is self-describing and cannot be silently paired with the
    wrong row after a CSV is regenerated.
    """
    if token_ids is None:
        return None
    path = _generated_tokens_dir(output_csv) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "generated_token_ids": [int(t) for t in token_ids]}
    payload.update(provenance or {})
    path.write_text(json.dumps(payload))
    return path


def _exact_token_window(tokenizer, ids: list[int], prompt_len: int,
                        max_probe: int = 8, max_shift: int = 4096):
    """Pick the source length whose re-encoded prompt is EXACTLY `prompt_len`.

    Returns `(text, got, source_len)`.

    The engine takes a prompt *string* and re-tokenises it, so the ids the model
    sees are `encode(decode(ids[:L]))`, which need not be `ids[:L]` and need not
    even have the same length. Measured on the staged pool this is
    document-dependent: 9 of 10 books round-trip exactly at every rung, and one
    loses a constant 7 tokens at every length (130047 -> 130040), because a
    single token early in the book decodes to text that re-encodes to 7 tokens.

    The prompt length is not negotiable — it must be exactly
    `context - gen - 1` tokens for the sequence to be exactly the rung — so the
    source length `L` is adjusted by the observed deficit, which is locally
    constant. Two probes suffice in practice; the loop exists for the case where
    moving `L` crosses another such token.
    """
    if not ids:
        raise ValueError("empty source")
    L = min(prompt_len, len(ids))
    for _ in range(max_probe):
        if not 0 < L <= len(ids):
            break
        text = tokenizer.decode(ids[:L])
        got = tokenizer.encode(text, add_special_tokens=False)
        if len(got) == prompt_len:
            return text, got, L
        # `len(got) == L + deficit`, so moving L by the deficit removes it.
        L = L + (prompt_len - len(got))
        if abs(L - prompt_len) > max_shift:
            break
    raise RuntimeError(
        f"could not find a source length whose engine prompt is exactly "
        f"{prompt_len} tokens (last tried L={L}, engine prompt "
        f"{len(got) if 'got' in dir() else 'n/a'}); refusing to run at a context "
        f"that was not measured"
    )


def _prompt_gen_tokens(run: dict) -> int:
    """The generation length that defines the PROMPT window for this run.

    Every arm of a rung must be given the SAME prompt, because the plan's
    losslessness check compares token ids between a speculative run and its
    target-only partners. The prompt window is `[0, C - gen_tokens - 1)`, so if
    each arm used its own `max_new_tokens` the 128-token baselines would get a
    prompt 896 tokens longer than the 1024-token speculative run they are
    supposed to check -- a different context, over which token equality says
    nothing. The pairing guard would then report every one of those seven cells
    as BAD_PAIR, and a stage that declares `losslessness: required` would fail
    on a defect that only the prompt builder could have caused.

    A level therefore sets `prompt_gen_tokens` to the RUNG's generation length
    and leaves it equal to `max_new_tokens` for every other arm. Absent, the
    behaviour is exactly as before.
    """
    return int(run.get("prompt_gen_tokens") or run.get("max_new_tokens", 1024))


def _build_pg19_document_prompt(documents_json: str, context_length: int,
                                doc_id: str, tokenizer, gen_tokens: int = 1024):
    """Build one rung's prompt from a single PG-19 book.

    Returns `(prompt_text, continuation_ids, provenance)`.

    The document is one book, not an offset into a concatenated stream, so the
    plan's unit of independence is real. Windows, from one contiguous span:

        prompt        [0, C - gen_tokens - 1)
        continuation  [C - gen_tokens - 1, C - 1)

    The `- 1` is the engine's leading BOS: `generate_text` calls the tokenizer
    without `add_special_tokens=False`, so the model sees one token more than
    the prompt ids counted here. Without it the sequence would be C + 1, one
    position past the native 128k window, which is exactly the kind of silent
    off-by-one that turns a rung into an extrapolation measurement.

    The continuation is what the runs generate into and what perplexity is
    scored on. It has to be contiguous with the prompt: a continuation placed
    after a generation-length gap would need a forward of more than C, which at
    native 128k would again measure extrapolation rather than the rung.
    """
    import numpy as np
    meta = json.loads(Path(documents_json).read_text())
    docs = {d["doc_id"]: d for d in meta["documents"]}
    if doc_id not in docs:
        raise ValueError(
            f"{documents_json}: unknown doc_id {doc_id!r}; "
            f"available: {sorted(docs)[:5]}..."
        )
    d = docs[doc_id]
    prompt_len = context_length - gen_tokens - 1
    if prompt_len < 1:
        raise ValueError(
            f"context {context_length} leaves no room for a prompt after "
            f"{gen_tokens} generated tokens and the leading BOS"
        )
    if d["length"] < context_length:
        raise RuntimeError(
            f"{doc_id}: {d['length']} tokens, need {context_length} for "
            f"context {context_length} (prompt {prompt_len} + continuation "
            f"{gen_tokens})"
        )
    arr = np.memmap(d["file"], dtype="int32", mode="r")
    source_len = min(d["length"], prompt_len + MAX_PROMPT_SHIFT + 1024)
    text, got, used = _exact_token_window(
        tokenizer, arr[:source_len].astype(int).tolist(), prompt_len)
    # Contiguous with the PROMPT TEXT: the continuation is the book text that
    # follows the prompt the model actually sees, so perplexity is conditioned
    # on the passage it scores, and the losslessness pairing is against the same
    # prompt.
    cont = arr[used:used + gen_tokens].astype(int).tolist()
    if len(cont) != gen_tokens:
        raise RuntimeError(
            f"{doc_id}: continuation is {len(cont)} tokens, need {gen_tokens}; "
            f"the document is too short for this rung"
        )
    provenance = {
        "doc_id": doc_id,
        # The exact ids the engine will feed. Saved so a second implementation
        # can be given THIS prompt rather than one rebuilt from the same book:
        # a rebuilt prompt is a different prompt, and an acceptance or
        # throughput comparison against it is not like-for-like.
        "prompt_token_ids": [int(t) for t in got],
        "doc_title": d.get("title"),
        "doc_url": d.get("url"),
        "prompt_tokens": len(got),
        "prompt_sha256": hashlib.sha256(
            ",".join(map(str, got)).encode()).hexdigest(),
        "continuation_sha256": hashlib.sha256(
            ",".join(map(str, cont)).encode()).hexdigest(),
        # Sequence the engine will actually build: prompt + BOS + generation.
        "sequence_tokens": len(got) + 1 + gen_tokens,
        "prompt_source_len": int(used),
    }
    sequence_tokens = len(got) + 1 + gen_tokens      # + 1 for the leading BOS
    if sequence_tokens != context_length:
        # Not ">": an under-length sequence is also wrong, because the rung is
        # supposed to BE that context length. Both directions are a measurement
        # of a different context than the one recorded.
        raise RuntimeError(
            f"{doc_id}: the engine would build a {sequence_tokens}-token "
            f"sequence ({len(got)} prompt + 1 BOS + {gen_tokens} generated) for "
            f"rung context {context_length}; refusing to run at a context we "
            f"would not be measuring"
        )
    return text, cont, provenance


def _measure_target_ppl(engine, prompt: str, continuation_ids: list[int],
                        world_size: int, local_rank: int):
    """Perplexity of the target on a held-out continuation after the prompt.

    Returns `(ppl, scored_tokens)` on every rank (the all-reduce is inside), so
    the caller does not have to reason about which rank holds the total.

    The sequence is encoded exactly as `generate_text` encodes it, i.e. with the
    tokenizer's default special tokens, so the count that reaches the model here
    is the same one the generation used. Scoring only the continuation means the
    prompt contributes context without contributing loss.
    """
    import torch
    import torch.distributed as dist
    from src.analysis.target_quality import continuation_nll, perplexity_from_sums

    model = engine.target_model
    prompt_ids = engine.tokenizer(prompt)["input_ids"]
    ids = torch.tensor([list(prompt_ids) + list(continuation_ids)],
                       dtype=torch.long, device=model.device)
    n_total = int(ids.shape[1])
    if world_size > 1 and n_total % world_size != 0:
        raise RuntimeError(
            f"target-quality sequence {n_total} is not divisible by "
            f"world_size {world_size}; shard bounds would not match the "
            f"engine's contiguous layout"
        )

    def _forward(local_ids, abs_pos):
        return model.model(
            input_ids=local_ids, position_ids=abs_pos, use_cache=False,
            past_key_values=None,
        ).last_hidden_state

    total, n, _ = continuation_nll(
        model, ids, score_from=len(prompt_ids), forward=_forward,
        rank=local_rank, world_size=world_size,
    )
    if world_size > 1:
        buf = torch.tensor([total, float(n)], dtype=torch.float64,
                           device=model.device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        total, n = float(buf[0]), int(buf[1])
    return perplexity_from_sums(total, n), n


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
            # MLSys analysis plan: the document is the unit of independence, so
            # a level may name a list of documents and expand into one run per
            # document. Seeds are NOT an expansion axis for these runs — greedy
            # decoding makes a seed change a no-op — but the suffix is kept so
            # run ids stay unique and sortable against the older rows.
            documents = level.get("documents")
            for seed in seeds:
                run = deepcopy(defaults)
                run.update({k: v for k, v in level.items() if k not in ("notes",)})
                run["seed"]     = seed
                run["group"]    = group_id
                run["level_id"] = level["id"]
                run["debug"]    = debug
                if documents:
                    for doc_id in documents:
                        r = deepcopy(run)
                        r["doc_id"] = doc_id
                        r["run_id"] = f"{level['id']}_{doc_id}_s{seed}"
                        runs.append(r)
                else:
                    run["run_id"] = f"{level['id']}_s{seed}"
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


def format_dry_run(runs: list[dict]) -> str:
    """The dry-run table, as text. THE authoritative printer.

    `scripts/mlsys_manifest.sh` counts the run lines of this output to decide how
    many rows a stage must write (`expected_rows`), and the rehearsal stub prints
    it instead of a copy of it. A second printer that drifted by a character in
    the separator would make the manifest's row check agree with the stub and
    with nothing else -- the check would pass while the real run's plan was never
    looked at.
    """
    lines = [f"\n{'RUN ID':<35}  {'GROUP':<5}  {'SEED':>5}  CONFIG", "-" * 80]
    for r in runs:
        config_summary = (
            f"draft={r['draft_model_name'].split('/')[-1]}  "
            f"k={r['spec_steps']}  "
            f"block={r['kv_block_size']}  "
            f"prefetch={r['prefetch_depth']}  "
            f"target={r['target_model_name'].split('/')[-1]}  "
            f"dwindow={r.get('draft_window_cap') or 'native'}  "
            f"ddtype={r.get('draft_dtype', 'auto')}"
        )
        lines.append(
            f"{r['run_id']:<35}  {r['group']:<5}  {r['seed']:>5}  {config_summary}")
    lines.append(f"\n{len(runs)} runs total.")
    return "\n".join(lines)


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

def build_prompt(context_length: int, tokenizer,
                 source: str = "synthetic",
                 pg19_meta: str | None = None,
                 seed: int = 42,
                 ruler_sidecar_dir: str | None = None,
                 run_id: str | None = None,
                 margin: int = 0) -> str:
    """Build a prompt of approximately `context_length` tokens.

    source="synthetic" (default): repeated technical-English paragraph.
      Conservative baseline — highly repetitive, which historically
      gives lower draft-target acceptance because the model is
      uncertain about whether to continue the pattern.

    source="pg19": real PG-19 narrative chunk picked by `seed`. Used
      by p35d to test whether acceptance recovers under natural text.
      Requires --prompt-pg19-meta to point at a preprocess_pg19.py
      metadata.json.

    source="ruler_niah": RULER-style needle-in-haystack — filler text
      with a seed-derived magic number embedded at a seeded position
      and a question appended at the end. Writes the expected answer +
      needle position to a sidecar JSON for post-hoc scoring (p38).
    """
    if source == "pg19":
        if pg19_meta is None:
            raise ValueError("source='pg19' requires pg19_meta path")
        return _build_pg19_prompt(context_length, tokenizer, pg19_meta, seed,
                                  margin=margin)
    if source == "ruler_niah":
        return _build_ruler_niah_prompt(
            context_length, tokenizer, seed,
            sidecar_dir=ruler_sidecar_dir, run_id=run_id,
        )
    # Default: synthetic repeated-paragraph.
    # A7 — the sentence ORDER is derived from `seed`, so three seeds produce
    # three genuinely different prompts (different token ids), not one prompt
    # decoded three times. Before this the builder ignored `seed` entirely
    # and "3 seeds" was really "1 prompt x 3 decoding seeds".
    #
    # NOTE: Arm1 and Arm2 share this TEMPLATE but not the same token ids —
    # they use different tokenizers (Llama-2 BPE vs the Llama-3 tiktoken-style
    # vocabulary), so even the same seed yields different token sequences
    # across arms. Cross-arm prompt identity was never achievable; what is
    # required is that each arm's seeds differ from each other.
    _SYNTHETIC_SENTENCES = (
        "The following is a detailed technical analysis of distributed machine learning systems. ",
        "Ring attention enables long-context inference by sharding the sequence across GPUs. ",
        "Speculative decoding accelerates generation by using a smaller draft model. ",
        "Key-value cache compression reduces the memory footprint of long sequences. ",
        "Rotary position embeddings interpolate smoothly across extended contexts. ",
        "Pipeline parallelism partitions layers while tensor parallelism splits weights. ",
        "Memory bandwidth, not compute, usually bounds decoding throughput. ",
        "Interconnect latency determines how well ring rotation overlaps with attention. ",
    )
    rng = random.Random(int(seed))
    order = list(_SYNTHETIC_SENTENCES)
    rng.shuffle(order)
    block = "".join(order)
    # add_special_tokens=False is REQUIRED: this list is repeated below, so a
    # BOS prepended here is replicated at every repetition boundary. With the
    # default it appeared 1337 times (once per ~97 tokens, 4 in the last 400),
    # which asks the model to continue a document that restarts continuously.
    # That is what collapsed ARM4-f1 acceptance to 0.24 from round 0, while
    # Arm2 — whose builder repeated a STRING and encoded once — measured 0.89.
    block_tokens = tokenizer.encode(block, add_special_tokens=False)
    if not block_tokens:
        block_tokens = tokenizer.encode(
            "The following is a detailed technical analysis. ",
            add_special_tokens=False)
    # Over-provision so we can trim DOWN to hit the target exactly. The
    # engine takes a prompt STRING and re-tokenises it, so the id count can
    # drift across decode->encode; B2 requires prompt+output to fit inside
    # the native window, so we converge on the requested count rather than
    # hoping the round-trip is lossless.
    #
    # The round trip is LOSSY here and NOT merely off by a constant: each
    # repetition re-merges the sentence-boundary tokens, so re-encoding
    # returns roughly one token fewer per block. Feeding decode() exactly
    # `context_length` ids therefore yields a prompt SHORTER than requested
    # (measured: 130944 -> 129595), and a fixed over-provision cannot fix that
    # because the loss grows with the repetition count. So solve for the id
    # count whose ENGINE-VISIBLE length equals the target, using the observed
    # ratio to converge in a few encodes instead of a linear scan (each
    # encode is ~130k tokens, so a scan would be minutes).
    reps = context_length // max(1, len(block_tokens)) + 64
    pool = block_tokens * reps
    n = context_length
    text = tokenizer.decode(pool[:n])
    L = len(tokenizer.encode(text, add_special_tokens=False))
    for _ in range(14):
        if L == context_length:
            break
        n = max(1, int(round(n * context_length / max(1, L))))
        if L < context_length:
            n += 1
        if n > len(pool):
            break
        text = tokenizer.decode(pool[:n])
        L = len(tokenizer.encode(text, add_special_tokens=False))
    if L != context_length:
        # Local scan to land exactly; the ratio step can oscillate by one.
        for delta in list(range(1, 96)) + list(range(-1, -96, -1)):
            m = n + delta
            if m < 1 or m > len(pool):
                continue
            t = tokenizer.decode(pool[:m])
            l2 = len(tokenizer.encode(t, add_special_tokens=False))
            if l2 == context_length:
                n, text, L = m, t, l2
                break
    if L != context_length:
        logger.warning(
            "synthetic prompt length %d != requested %d after convergence",
            L, context_length)
    return text


def _build_ruler_niah_prompt(context_length: int, tokenizer, seed: int,
                              sidecar_dir: str | None = None,
                              run_id: str | None = None) -> str:
    """RULER-style needle-in-haystack at the given context length.

    Structure (token-budget aware):
        [filler]   The magic number is <N>. Remember this number.   [more filler]
        What is the magic number? The magic number is

    The needle's magic number and position are derived from `seed` so
    re-running with the same seed produces an identical prompt. Both
    values are written to a sidecar JSON (one per run_id) so the
    post-hoc scorer (scripts/score_ruler_niah.py) can mark accuracy
    without re-deriving them.

    Returns the decoded prompt string.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    magic_number = int(rng.integers(10_000, 99_999))
    needle_text = f" The magic number is {magic_number}. Remember this number well. "
    question_text = " What is the magic number? The magic number is"
    filler_base = (
        "The river flowed steadily through the valley. Many farmers tended the land. "
        "The sun rose each morning over the hills. People worked through the day. "
        "Trade routes connected distant villages. Goods moved across the countryside. "
    )
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    question_ids = tokenizer.encode(question_text, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler_base, add_special_tokens=False)

    # Total budget for filler around the needle. Reserve a bit at the end
    # for the question so the model sees it last.
    filler_budget = max(
        0,
        context_length - len(needle_ids) - len(question_ids),
    )
    # Position needle at a seeded fraction of the filler region. We bound
    # it away from the extreme ends (5%-95%) so the test isn't dominated
    # by recency or primacy effects.
    needle_frac = float(rng.uniform(0.05, 0.95))
    pre_len = int(filler_budget * needle_frac)
    post_len = filler_budget - pre_len

    def _stretch(target_len: int) -> list:
        if target_len <= 0:
            return []
        reps = (target_len // len(filler_ids)) + 1
        return (filler_ids * reps)[:target_len]

    full_ids = (
        _stretch(pre_len) + needle_ids + _stretch(post_len) + question_ids
    )
    full_ids = full_ids[:context_length]

    if sidecar_dir is not None and run_id is not None:
        sidecar_path = Path(sidecar_dir) / f"{run_id}.ruler_niah.json"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps({
            "run_id": run_id,
            "seed": seed,
            "context_length": context_length,
            "magic_number": magic_number,
            "needle_position_frac": needle_frac,
            "needle_position_tokens": pre_len,
            "needle_text": needle_text.strip(),
            "question_text": question_text.strip(),
        }, indent=2))

    return tokenizer.decode(full_ids)


def _build_pg19_prompt(context_length: int, tokenizer,
                       meta_path: str, seed: int, margin: int = 0) -> str:
    """Load a PG-19 chunk slice of context_length tokens and decode it.

    Picks a chunk pseudorandomly seeded from `seed` (same logic as
    eval_perplexity_matrix._load_pg19_chunk), then decodes back to
    text so the existing prompt-as-string flow stays unchanged.
    """
    import numpy as np
    meta = json.loads(Path(meta_path).read_text())
    chunks = meta["chunks"]
    if not chunks:
        raise RuntimeError(f"{meta_path}: no chunks in metadata")
    # Long-context gate/natural-text runs need prompt + a scored continuation
    # inside ONE chunk, so prefer the longest chunks and take a contiguous
    # slice. `margin` is the extra tokens the caller wants held back after the
    # prompt; it defaults to 0 for the historical dose-response callers.
    need = context_length + int(margin)
    suitable = sorted((c for c in chunks if c["length"] >= need),
                      key=lambda c: c["file"])
    rng = np.random.default_rng(seed)
    if suitable:
        c = suitable[int(rng.integers(0, len(suitable)))]
        arr = np.memmap(c["file"], dtype="int32", mode="r")
        start = int(rng.integers(0, c["length"] - need + 1))
        ids = list(arr[start:start + context_length].astype(int))
    else:
        # Concatenate chunks if no single chunk is long enough.
        #
        # WARNING (not exercised by any published cell): stitching chunks
        # concatenates independently-tokenised id arrays, so every chunk
        # boundary carries whatever special token its preprocessing prepended
        # (a BOS, typically). That would inject one BOS per boundary — the
        # same failure mode as the synthetic-prompt BOS bug, though via a
        # different route. Any caller reaching here MUST strip boundary
        # specials before concatenating, or stage longer chunks instead; the
        # long-context gate stages 1M-token chunks precisely to avoid it.
        joined = []
        for c in chunks:
            arr = np.memmap(c["file"], dtype="int32", mode="r")
            joined.extend(arr.astype(int).tolist())
            if len(joined) >= context_length:
                break
        ids = joined[:context_length]
    text = tokenizer.decode(ids)
    # The engine takes a prompt STRING and re-tokenises it, so verify the
    # caller actually gets the requested token count. Natural text round-trips
    # far more cleanly than the synthetic repetition (which loses ~1 token per
    # block), but "cleanly" is not "exactly", and a silently short prompt would
    # break B2 (prompt+output inside the native window).
    got = len(tokenizer.encode(text, add_special_tokens=False))
    if got != context_length:
        log.warning(
            "PG-19 prompt round-trip: requested %d tokens, engine will see %d "
            "(delta %+d, %.4f%%)", context_length, got, got - context_length,
            100.0 * (got - context_length) / max(1, context_length))
    return text



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


def decode_rate_tps(tokens_generated, time_sec, ttft_ms) -> float:
    """Decode-only tokens/second: `(tokens_generated - 1) / (time_sec - ttft)`.

    The first token is the product of prefill, not of the decode loop: it is
    already available at ttft. The remaining `tokens_generated - 1` tokens are
    what the post-prefill wall actually produced, so dividing the FULL token
    count by the decode wall overstates the decode rate by one token's worth.

    Both arms use this same helper, because the paired ratio is a comparison of
    two decode rates: computing it one way for the spec arm and another for the
    target-only arm would put the difference into the ratio.
    """
    n = int(tokens_generated)
    if n < 1:
        return 0.0
    wall = max(float(time_sec) - (ttft_ms or 0.0) / 1000.0, 1e-9)
    return (n - 1) / wall


def _arm_role(run: dict) -> str:
    """What this row is, for pairing: spec, target_full or target_short.

    Taken from the group name when the config supplies one (the stage configs
    name their groups `*_TARGET_FULL` / `*_TARGET_SHORT`), and otherwise from the
    generation length against the spec arm's: a target-only row that generates
    the full 1024 tokens is the short arm's complement, not a different thing.
    """
    group = str(run.get("group", "") or "").upper()
    if int(run.get("spec_steps", 0) or 0) > 0:
        return "spec"
    if group.endswith("TARGET_FULL") or "TARGET_FULL" in group:
        return "target_full"
    if group.endswith("TARGET_SHORT") or "TARGET_SHORT" in group:
        return "target_short"
    return f"target_{int(run.get('max_new_tokens', 0) or 0)}"


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
        #
        # timeout=1 hour: the default NCCL watchdog timeout is 10 minutes,
        # but at 1M context the prefill itself takes ~20-25 min and a
        # single ring-attention coalesced op can block long enough for
        # the slowest rank to trip the watchdog. Bumping to 1 hr is
        # plenty even at 1M and still fails fast on real deadlocks.
        # (Phase C 2026-05-10 first 1M attempt died at SeqNum=36 after
        # 600s with rank 7's ring rotation hung waiting for rank 2.)
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(hours=1),
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
            # MLSys B1 — explicit extrapolation factor (None = automatic).
            rope_factor       = (float(run["rope_factor"])
                                 if run.get("rope_factor") is not None else None),
            # MLSys — YaRN anchor override (None = the model's own window).
            # Set to the model's true pretraining base (8192 for Llama-3.1)
            # to build on the native scaling instead of re-basing it.
            rope_anchor_base  = (int(run["rope_anchor_base"])
                                 if run.get("rope_anchor_base") is not None
                                 else None),
            # MLSys B3 — exact-length generation (EOS ignored).
            ignore_eos        = bool(run.get("ignore_eos", False)),
            # C11 NF4 KV-cache. Default False -> M3 byte-identical.
            # Phase C P3.5 final matrix uses kv_quant=true at all contexts.
            kv_quant          = bool(run.get("kv_quant", False)),
            # MLSys Phase 1/3 — draft isolation knobs. Both default to
            # None/"auto", which reproduces M3/M4 behaviour exactly.
            draft_window_cap  = (int(run["draft_window_cap"])
                                 if run.get("draft_window_cap") else None),
            draft_dtype       = str(run.get("draft_dtype", "auto")),
            # M4 Phase C 2026-05-10 NF4 acceptance-recovery levers. The
            # YAMLs don't override these by default; we let the RASDConfig
            # defaults apply (kv_outlier_prefix_size=128, kv_block_size_nf4=32).
            # Wired through here so an experiment can A/B test by setting
            # them in the YAML if needed.
            kv_outlier_prefix_size = int(run.get("kv_outlier_prefix_size", 128)),
            kv_block_size_nf4      = int(run.get("kv_block_size_nf4", 32)),
            nf4_update_chunk_size  = int(run.get("nf4_update_chunk_size", 2048)),
            # M4 paper memory attribution. Off by default. When True,
            # rasd_inference.generate() snapshots per-stage GPU memory
            # via MemoryTracer and writes a JSON sidecar.
            memory_trace           = bool(run.get("memory_trace", False)),
            memory_trace_dir       = run.get("memory_trace_dir") or None,
            seed              = int(run["seed"]),
            debug             = bool(run.get("debug", False)),
            # C13 per-position trace (M4 Phase A2). Default off so M3
            # replay is byte-identical when --log-per-token is unset.
            log_per_token     = bool(run.get("log_per_token", False)),
            # MLSys analysis plan (a) — return raw generated token IDs so the
            # losslessness check can compare token streams, not decoded text.
            save_generated_tokens = bool(run.get("save_generated_tokens", False)),
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
        prompt_source = run.get("prompt_source", "synthetic")
        continuation_ids: list[int] = []
        doc_prov: dict = {}
        if prompt_source == "pg19_document":
            doc_json = run.get("prompt_documents_json")
            if not doc_json:
                raise ValueError(
                    "prompt_source='pg19_document' requires prompt_documents_json"
                )
            doc_id = run.get("doc_id")
            if not doc_id:
                raise ValueError(
                    "prompt_source='pg19_document' requires doc_id"
                )
            prompt, continuation_ids, doc_prov = _build_pg19_document_prompt(
                doc_json,
                int(run.get("context_length", 65536)),
                doc_id,
                engine.tokenizer,
                gen_tokens=_prompt_gen_tokens(run),
            )
        else:
            prompt = build_prompt(
                int(run.get("context_length", 65536)),
                engine.tokenizer,
                source=prompt_source,
                pg19_meta=run.get("prompt_pg19_meta"),
                seed=int(run.get("seed", 42)),
                ruler_sidecar_dir=run.get("ruler_sidecar_dir"),
                run_id=run["run_id"],
            )

        # MLSys A6 — per-row prompt provenance. Record the exact prompt
        # token count and a content hash so a result can be tied to the
        # precise input that produced it (A7 makes the prompt seed-dependent,
        # so this is the only way to prove the three seeds really differ).
        try:
            _pids = engine.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            row["prompt_tokens"] = len(_pids)
            row["prompt_sha256"] = hashlib.sha256(
                ",".join(str(i) for i in _pids).encode()).hexdigest()
        except Exception as _e:  # noqa: BLE001
            row["prompt_tokens"] = ""
            row["prompt_sha256"] = ""
            log.warning("prompt provenance failed: %r", _e)
        row["target_revision"] = run.get("target_revision") or ""
        row["draft_revision"] = run.get("draft_revision") or ""
        row["prompt_source"] = prompt_source
        # PLANNED sequence length, from the document pool: prompt + 1 BOS + the
        # RUNG's generation length. Correct for an arm that generates the rung's
        # full length; overwritten below with what the engine actually built,
        # because the two differ by design for the 128-token baselines (a shorter
        # generation into the same prompt) and differ by accident whenever a run
        # stops early.
        if doc_prov.get("sequence_tokens") is not None:
            row["sequence_tokens"] = doc_prov["sequence_tokens"]
        row["doc_id"] = run.get("doc_id", "") or ""
        # pair_id: rung + document, shared across the arms compared. The rung is
        # in the key because the same document is run at several contexts and a
        # document-only key would pair a 128k spec row with a 256k baseline.
        _ctx = run.get("context_length", "")
        row["pair_id"] = f"{_ctx}:{row['doc_id']}" if row["doc_id"] else ""
        row["arm_role"] = _arm_role(run)
        row["rope_arm"] = str(run.get("rope_arm", "") or "")
        row["temperature"] = run.get("temperature", 1.0)
        row["top_p"] = run.get("top_p", 1.0)
        row["ignore_eos"] = bool(run.get("ignore_eos", False))
        if doc_prov:
            # The pool builder hashed the same token IDs this row just hashed.
            # If they disagree the row is not the document it claims to be, and
            # pairing it with its target-only counterpart would be unsound.
            if doc_prov["prompt_sha256"] != row["prompt_sha256"]:
                log.warning(
                    "document prompt hash mismatch for %s: pool=%s engine=%s",
                    doc_prov["doc_id"], doc_prov["prompt_sha256"],
                    row["prompt_sha256"],
                )

        # M4 C7 — torch.profiler wrap (default off). Enabled per-row via
        # run["profile"] = True (set from --profile CLI flag below).
        profile_enabled = bool(run.get("profile", False)) and local_rank == 0
        if profile_enabled:
            from src.analysis.profiler import RoundProfiler
            with RoundProfiler(enabled=True) as _prof:
                generated_text, metrics = engine.generate_text(prompt)
            metrics["_profiler_summary"] = _prof.summary
        else:
            generated_text, metrics = engine.generate_text(prompt)

        # RULER niah: write generated text to the sidecar dir so the
        # post-hoc scorer can mark accuracy without having to re-run.
        if (local_rank == 0
                and run.get("prompt_source") == "ruler_niah"
                and run.get("ruler_sidecar_dir")):
            ruler_dir = Path(run["ruler_sidecar_dir"])
            ruler_dir.mkdir(parents=True, exist_ok=True)
            (ruler_dir / f"{run['run_id']}.generated.txt").write_text(
                generated_text or ""
            )

        # M4 Phase D 2026-05-11 (F5 qualitative table source): when
        # --save-generated-text is on, write the decoded generation to
        # <output_csv_dir>/generated/<run_id>.txt for every run. Used
        # by the side-by-side qualitative comparison table.
        if (local_rank == 0
                and run.get("save_generated_text", False)
                and generated_text is not None):
            gen_dir = Path(output_csv).resolve().parent / "generated"
            gen_dir.mkdir(parents=True, exist_ok=True)
            (gen_dir / f"{run['run_id']}.txt").write_text(generated_text)

        # Pull the per-position trace out of the metrics dict before
        # wandb logging — it's a list-of-dicts, not a wandb-loggable
        # scalar. Only rank 0 receives a non-None trace (others get
        # None per RASDInference.generate's rank-0 guard).
        gen_ids = metrics.pop("generated_token_ids", None)
        # The ids the TARGET was actually fed, BOS included, from the engine's
        # own tensor. `prompt_tokens` and `prompt_sha256` are the prompt WITHOUT
        # the BOS -- that is what the engine's prompt builder is defined over --
        # so a consumer that claims to replay the target's input needs this
        # field, and comparing it against the no-BOS ids is a one-token
        # difference at the front of every rotary phase.
        engine_input_ids = metrics.pop("engine_input_ids", None)
        if engine_input_ids is None:
            try:
                # The same call generate_text makes: the engine's real input.
                engine_input_ids = engine.tokenizer(
                    prompt, add_special_tokens=True)["input_ids"]
            except Exception:                       # noqa: BLE001
                engine_input_ids = None
        trace = metrics.pop("per_token_trace", None)
        prof_summary = metrics.pop("_profiler_summary", None)
        if local_rank == 0:
            tok_sidecar = write_generated_tokens_sidecar(
                output_csv, run["run_id"], gen_ids, {
                    **doc_prov,
                    "engine_input_ids": engine_input_ids,
                    "prompt_tokens": row.get("prompt_tokens"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "context_length": run.get("context_length"),
                    "max_new_tokens": run.get("max_new_tokens"),
                    "spec_steps": run.get("spec_steps"),
                    "temperature": run.get("temperature", 1.0),
                    "top_p": run.get("top_p", 1.0),
                    "ignore_eos": bool(run.get("ignore_eos", False)),
                })
            if tok_sidecar is not None:
                log.info("Wrote generated token ids: %s (%d tokens)",
                         tok_sidecar, len(gen_ids))
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

        # Target quality beside acceptance (plan 4.3). Measured after the
        # generation so a quality number always ships with its acceptance
        # number; an acceptance figure alone cannot distinguish a healthy
        # target from a broken one.
        if run.get("measure_target_ppl") and continuation_ids:
            try:
                ppl, n_ppl = _measure_target_ppl(
                    engine, prompt, continuation_ids, world_size, local_rank,
                )
                row["target_ppl"] = round(ppl, 6)
                row["ppl_tokens"] = n_ppl
            except Exception as _pe:  # noqa: BLE001
                # A failed quality measurement must not discard an otherwise
                # good generation, but it must be visible rather than blank.
                row["target_ppl"] = ""
                row["ppl_tokens"] = ""
                row["error"] = f"target_ppl failed: {type(_pe).__name__}: {_pe}"
                log.warning("target_ppl failed: %r", _pe)
        row["generated_tokens_sha256"] = (
            hashlib.sha256(",".join(str(i) for i in gen_ids).encode()).hexdigest()
            if gen_ids else ""
        )
        # Decode-only rate. The first token arrives at ttft, so the remaining
        # `tokens_generated - 1` tokens are the post-prefill wall's product;
        # either convention is defensible and both arms use this one.
        # The sequence the engine actually held, as the ENGINE measured it --
        # `int(generated_ids.shape[1])`, the length of the final sequence tensor.
        #
        # It is deliberately NOT `prompt_tokens + 1 + tokens_generated`. That
        # expression is the identity the row is supposed to be checked AGAINST,
        # so writing it here made the check a restatement of the row's own
        # arithmetic: it could not fail, and it verified nothing. Taken from the
        # engine, the check compares three independently obtained numbers -- the
        # prompt length (from the tokenizer), the emitted ids (from the
        # sidecar) and this one (from the tensor).
        seq = metrics.get("sequence_tokens")
        if seq is None:
            # An engine that does not report it leaves the planned value in
            # place, which the identity check will reject. Inventing a number
            # here is what the check exists to prevent.
            log.warning("engine reported no sequence_tokens for %s; the identity "
                        "check will fail this row", run["run_id"])
        else:
            row["sequence_tokens"] = int(seq)
        row.update({
            "tokens_generated": metrics["tokens_generated"],
            "time_sec":         round(metrics["time_sec"], 4),
            "throughput_tps":   round(metrics["throughput_tps"], 2),
            "decode_tps":       round(decode_rate_tps(metrics["tokens_generated"],
                                                      metrics["time_sec"],
                                                      metrics.get("ttft_ms")), 3),
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
        # Capture the full traceback (not just str(exc)) so error rows in
        # the CSV are actionable. Without this, the parent only saw the
        # exception message — the ctx512k/ctx1M target-only failures on
        # 2026-05-10 lost all stack-trace context to a wandb buffer drop.
        import traceback as _tb
        tb_str = _tb.format_exc()
        # Single-line for CSV-friendliness; still includes file:line markers.
        row["error"] = f"{type(exc).__name__}: {exc} | {tb_str.replace(chr(10), ' || ')}"
        # Also dump the full multi-line traceback to a per-run sidecar so
        # the next debugger doesn't have to un-mangle the CSV escaping.
        if local_rank == 0:
            err_dir = Path(output_csv).resolve().parent / "errors"
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / f"{run['run_id']}.traceback.txt").write_text(tb_str)
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

def apply_cli_to_runs(args, runs: list[dict]) -> None:
    """CLI flags -> the run dicts. One function, so nothing can diverge.

    This block used to live inside `main()`, which meant the rehearsal's stub had
    to carry its own copy of the same logic. A copy of seven propagation rules is
    a place for the rehearsal and the real run to disagree -- and the whole point
    of the rehearsal is that they cannot. Both callers now share this, including:

      * `--profile` (the per-run torch profiler wrap),
      * prompt-source VALIDATION (`pg19` without `--prompt-pg19-meta` is a
        configuration error, not a silently synthetic prompt),
      * the RULER sidecar directory setup, which defaults to a path derived from
        `--output`.
    """
    # Propagate the --log-per-token flag onto every run dict so the
    # subprocess worker picks it up (each run is serialized via --_worker).
    if args.log_per_token:
        for r in runs:
            r["log_per_token"] = True
    if args.profile:
        for r in runs:
            r["profile"] = True
    if args.save_generated_text:
        for r in runs:
            r["save_generated_text"] = True
    if args.save_generated_tokens:
        for r in runs:
            r["save_generated_tokens"] = True
    if args.prompt_source != "synthetic":
        if args.prompt_source == "pg19" and not args.prompt_pg19_meta:
            raise SystemExit("--prompt-source=pg19 requires --prompt-pg19-meta")
        # Default RULER sidecar dir is <output_csv_dir>/ruler/
        ruler_dir = (args.ruler_sidecar_dir
                     or str(Path(args.output).resolve().parent / "ruler"))
        for r in runs:
            r["prompt_source"] = args.prompt_source
            if args.prompt_pg19_meta:
                r["prompt_pg19_meta"] = args.prompt_pg19_meta
            if args.prompt_source == "ruler_niah":
                r["ruler_sidecar_dir"] = ruler_dir
    if args.memory_trace:
        for r in runs:
            r["memory_trace"] = True
            # Default sidecar dir is alongside the output csv:
            #   <output_csv>/../memory_trace/<run_id>.rank<r>.json
            r.setdefault(
                "memory_trace_dir",
                str(Path(args.output).parent / "memory_trace"),
            )
    if args.checkpoint_every > 0:
        for r in runs:
            # Per-run override wins; CLI flag is a default for runs that
            # don't specify their own checkpoint_every in the YAML
            r.setdefault("checkpoint_every", args.checkpoint_every)
    # MLSys Phase 1/3 — draft isolation knobs. `setdefault` so a YAML level
    # can pin its own value (arm configs are one-YAML-per-arm, so both
    # routes work); the CLI flag acts as the grid-wide default.
    if args.draft_window_cap is not None:
        for r in runs:
            r.setdefault("draft_window_cap", args.draft_window_cap)
    if args.draft_dtype != "auto":
        for r in runs:
            r.setdefault("draft_dtype", args.draft_dtype)


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
    parser.add_argument("--stage-id", default=None,
                        help="Identifier of the stage that owns --output, used "
                             "to refuse writing a stage's results over an "
                             "unrelated artifact's filename (f).")
    parser.add_argument("--overwrite-stage", action="store_true",
                        help="Deliberately replace --output even if it belongs "
                             "to a different stage.")
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
    parser.add_argument("--memory-trace", action="store_true",
                        help="Per-rank GPU memory attribution snapshots at "
                             "generate() lifecycle points (post-load, post-prefill, "
                             "post-verify-rounds 1/2/4/8, end). Writes JSON sidecar "
                             "to <output_dir>/memory_trace/<run_id>.rank<r>.json. "
                             "Negligible overhead. Source for paper memory "
                             "attribution figure. Off by default so M3 replay "
                             "stays byte-identical.")
    parser.add_argument("--prompt-source", default="synthetic",
                        choices=["synthetic", "pg19", "ruler_niah"],
                        help="Prompt source for build_prompt(). 'synthetic' "
                             "(default) = repeated technical-English paragraph "
                             "(M3+M4 default). 'pg19' = real narrative text "
                             "from PG-19 chunks (p35d acceptance ablation). "
                             "'ruler_niah' = needle-in-haystack stress test "
                             "(p38 long-context capability eval).")
    parser.add_argument("--prompt-pg19-meta", default=None,
                        help="Path to pg19_<split>_metadata.json — required "
                             "when --prompt-source=pg19.")
    parser.add_argument("--ruler-sidecar-dir", default=None,
                        help="Directory to write per-run RULER needle metadata "
                             "JSONs (used by scripts/score_ruler_niah.py). "
                             "Defaults to <output_csv_dir>/ruler/.")
    parser.add_argument("--save-generated-tokens", action="store_true",
                        help="Write each run's raw generated token IDs to "
                             "<output_csv_dir>/tokens/<run_id>.json. "
                             "Losslessness is a token-level claim and decoded "
                             "text is not injective, so the IDs must survive "
                             "the run for the check to be possible at all. "
                             "Default off so M3 replay stays byte-identical.")
    parser.add_argument("--save-generated-text", action="store_true",
                        help="Write the decoded generated text from each run "
                             "to <output_csv_dir>/generated/<run_id>.txt. "
                             "Used by F5 (qualitative comparison table) and "
                             "F8 (low-acceptance error analysis). Default off "
                             "so M3 replay stays byte-identical.")
    # --- MLSys experiment program (2026-10-04) ---
    parser.add_argument("--draft-window-cap", type=int, default=None,
                        help="Hard cap (tokens) on the draft model's context "
                             "window. Default: the draft's native "
                             "max_position_embeddings (exact M3/M4 behaviour). "
                             "MLSys Phase 1 arm 2 uses 4096 to hold the draft "
                             "at a 4k window while the target runs natively at "
                             "128k. A YAML level may set its own "
                             "`draft_window_cap`, which takes precedence.")
    parser.add_argument("--draft-dtype", choices=["auto", "nf4", "bf16"],
                        default="auto",
                        help="Draft weight precision. 'auto' (default) follows "
                             "the config's quantize_draft — i.e. M3/M4 "
                             "behaviour. 'nf4' forces 4-bit NF4 (CUDA only). "
                             "'bf16' forces unquantized bf16 draft weights. "
                             "MLSys Phase 3 runs the same 64k cell both ways to "
                             "isolate NF4 draft/target logit divergence.")
    # Internal: subprocess worker mode
    parser.add_argument("--_worker",  default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ---- Subprocess worker path ----
    if args._worker:
        run = json.loads(args._worker)
        output_csv = args.output
        _guard_output_collision(output_csv, args.stage_id or "unnamed",
                                overwrite=args.overwrite_stage)
        _run_single_worker(run, args.wandb_project, output_csv)
        return

    cfg        = load_config(args.config)
    all_runs   = build_run_configs(cfg, args.groups, args.debug, seed_filter=args.seeds)
    apply_cli_to_runs(args, all_runs)
    output_csv = Path(args.output)

    log.info("Total runs: %d", len(all_runs))

    if args.dry_run:
        print(format_dry_run(all_runs))
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
            # 2026-05-10 fix: canary must inherit the same CLI-level
            # observability flags as the matrix runs (profile, per-token
            # trace, memory trace). Otherwise the canary silently skips
            # those sidecars and we can't catch issues like a missing
            # profile JSON until a 1-hr context run is half done.
            if args.profile:
                canary_run["profile"] = True
            if args.log_per_token:
                canary_run["log_per_token"] = True
            if args.save_generated_text:
                canary_run["save_generated_text"] = True
            if args.save_generated_tokens:
                canary_run["save_generated_tokens"] = True
            if args.memory_trace:
                canary_run["memory_trace"] = True
                canary_run.setdefault(
                    "memory_trace_dir",
                    str(Path(args.output).parent / "memory_trace"),
                )
            if args.draft_window_cap is not None:
                canary_run.setdefault("draft_window_cap", args.draft_window_cap)
            if args.draft_dtype != "auto":
                canary_run.setdefault("draft_dtype", args.draft_dtype)
            if args.prompt_source != "synthetic":
                canary_run["prompt_source"] = args.prompt_source
                if args.prompt_pg19_meta:
                    canary_run["prompt_pg19_meta"] = args.prompt_pg19_meta
                if args.prompt_source == "ruler_niah":
                    canary_run["ruler_sidecar_dir"] = (
                        args.ruler_sidecar_dir
                        or str(Path(args.output).resolve().parent / "ruler")
                    )
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
            # sys.exit(1) (not just `break`) so the calling stage in
            # phase_c_pod_session.sh sees a non-zero exit and the
            # master script halts. Same class of bug as the canary
            # path — a bare break + return propagates as Python exit 0,
            # which made the master script run p34/p35 against a
            # broken smoke output on 2026-05-10 attempt #4.
            sys.exit(1)

    # Summary
    import csv as _csv
    with open(output_csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    n_ok  = sum(1 for r in rows if r["status"] == "ok")
    n_err = sum(1 for r in rows if r["status"] != "ok")
    log.info("Done. %d succeeded, %d failed. Results: %s", n_ok, n_err, output_csv)


if __name__ == "__main__":
    main()

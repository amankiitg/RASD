#!/usr/bin/env python3
"""Phase D F6: aggregate Phase C CSVs into results/final/final_results.json.

Single canonical JSON that downstream figure scripts + LaTeX tables
read from. No re-computation — just normalization and indexing.

Sources:
  - results/final/final_matrix.csv        (p35 RASD spec mode)
  - results/final/target_only_matrix.csv  (p35b apples-to-apples baseline)
  - results/baselines/hf_ceiling.csv      (p37 vanilla HF FA-2 ceiling)
  - results/baselines/m4_baselines.csv    (p34 ring/sliding synthetic forwards)
  - results/perplexity/m4_ppl.csv         (p35c YaRN PPL)
  - results/perplexity/m4_ppl_vanilla_rope.csv (p35c vanilla baseline)
  - results/final/profiler_pass/profiler/*.json (p36 compute/comm/idle JSONs)
  - results/final/profiler_pass/profiler_pass.csv (p36 metric rows)
  - results/final/p35d_pg19_prompt.csv    (p35d PG-19 sanity)

Output schema (top-level keys):
  meta:           { phase: "C", commit_sha, timestamps }
  matrix:         { rasd: {ctx: [seed_rows]}, target_only: {...}, hf_ceiling: {...} }
  speedup:        { ctx: {rasd_mean, target_mean, ratio, n_seeds} }
  profiler:       { ctx: {compute_pct, comm_pct, idle_pct, other_pct, ...} }
  perplexity:     { yarn: [...], vanilla: [...] }
  pg19_prompt:    [p35d row]
  baselines:      { ring: {...}, sliding: {...} }
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x: str | None, default: float = 0.0) -> float:
    """Coerce a CSV cell to float, '' or '-1.0' → default."""
    if not x or x in ("-1.0", "-1"):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x: str | None, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _group_by_level(rows: list[dict], group_filter: str | None = None) -> dict:
    """Index matrix rows by level_id (context), filtered by group ('M4' usually)."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        if group_filter and r.get("group") != group_filter:
            continue
        if r.get("status") != "ok":
            continue
        level = r.get("level_id", "")
        if not level:
            continue
        out.setdefault(level, []).append({
            "run_id":           r["run_id"],
            "seed":             _i(r.get("seed")),
            "context_length":   _i(r.get("context_length")),
            "tokens_generated": _i(r.get("tokens_generated")),
            "time_sec":         _f(r.get("time_sec")),
            "throughput_tps":   _f(r.get("throughput_tps")),
            "acceptance_rate":  _f(r.get("acceptance_rate")),
            "mean_latency_ms":  _f(r.get("mean_latency_ms")),
            "ttft_ms":          _f(r.get("ttft_ms")),
            "gpu_peak_mem_mb":  _f(r.get("gpu_peak_mem_mb")),
            "n_rounds":         _i(r.get("n_rounds")),
        })
    return out


def _mean_std_ci(vals: list[float]) -> dict:
    """Mean + 95% CI half-width (t-distribution would be more correct
    for small n; we use ±1.96·SEM as a reasonable approximation since
    even at n=3 the qualitative shape is what matters)."""
    if not vals:
        return {"mean": None, "std": None, "ci_half": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "ci_half": 0.0, "n": 1}
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals)
    sem = sd / (len(vals) ** 0.5)
    return {
        "mean":     round(mu, 6),
        "std":      round(sd, 6),
        "ci_half":  round(1.96 * sem, 6),
        "n":        len(vals),
    }


def _per_ctx_stats(grouped: dict, metric: str) -> dict:
    """Compute mean/std/CI per context level for a given metric."""
    out: dict[str, dict] = {}
    for level, cells in grouped.items():
        vals = [c[metric] for c in cells if c.get(metric) is not None]
        out[level] = _mean_std_ci(vals)
        # Add ctx for downstream plotting
        if cells:
            out[level]["context_length"] = cells[0]["context_length"]
    return out


def build() -> dict:
    out = {
        "meta": {
            "phase":      "C",
            "commit_sha": _git_sha(),
            "generated":  datetime.now(timezone.utc).isoformat(),
            "version":    "1.0",
        },
        "matrix":     {},
        "speedup":    {},
        "memory":     {},
        "profiler":   {},
        "perplexity": {},
        "baselines":  {},
        "ablations":  {},
    }

    # --- p35 RASD matrix ---
    rasd_rows  = _read_csv(REPO_ROOT / "results/final/final_matrix.csv")
    rasd_g     = _group_by_level(rasd_rows, group_filter="M4")
    out["matrix"]["rasd"] = rasd_g

    # --- p35b target-only ---
    tgt_rows = _read_csv(REPO_ROOT / "results/final/target_only_matrix.csv")
    tgt_g    = _group_by_level(tgt_rows, group_filter="M4")
    out["matrix"]["target_only"] = tgt_g

    # --- p37 HF FA-2 ceiling (different schema: per-row ctx not level_id) ---
    hf_rows = _read_csv(REPO_ROOT / "results/baselines/hf_ceiling.csv")
    hf_out = []
    for r in hf_rows:
        hf_out.append({
            "context_length": _i(r.get("context_length")),
            "seed":           _i(r.get("seed")),
            "attn_impl":      r.get("attn_impl", ""),
            "status":         r.get("status", ""),
            "throughput_tps": _f(r.get("throughput_tps"), default=-1.0),
            "gpu_peak_mem_mb": _f(r.get("gpu_peak_mem_mb"), default=-1.0),
            "error":          r.get("error", "")[:120],
        })
    out["matrix"]["hf_ceiling"] = hf_out

    # --- Speedup table: RASD vs target-only, per ctx, 3-seed mean ---
    rasd_tps  = _per_ctx_stats(rasd_g, "throughput_tps")
    tgt_tps   = _per_ctx_stats(tgt_g,  "throughput_tps")
    for r_level, r_stat in rasd_tps.items():
        # Map e.g. M4_ctx128k -> TARGET_ctx128k
        t_level = r_level.replace("M4_ctx", "TARGET_ctx")
        t_stat  = tgt_tps.get(t_level)
        if not t_stat or t_stat["mean"] is None or r_stat["mean"] is None:
            continue
        out["speedup"][r_level] = {
            "context_length":     r_stat["context_length"],
            "rasd_mean_tps":      r_stat["mean"],
            "rasd_ci_half":       r_stat["ci_half"],
            "target_mean_tps":    t_stat["mean"],
            "target_ci_half":     t_stat["ci_half"],
            "ratio":              round(r_stat["mean"] / t_stat["mean"], 4),
            "n_seeds":            r_stat["n"],
        }

    # --- Memory per ctx (3-seed mean, peak GB) ---
    out["memory"]["rasd"]        = _per_ctx_stats(rasd_g, "gpu_peak_mem_mb")
    out["memory"]["target_only"] = _per_ctx_stats(tgt_g,  "gpu_peak_mem_mb")

    # --- p36 profiler JSONs (one per ctx, single seed=42) ---
    prof_dir = REPO_ROOT / "results/final/profiler_pass/profiler"
    if prof_dir.exists():
        for jf in sorted(prof_dir.glob("*.json")):
            level_id = jf.stem  # PROFILED_ctx128k_s42 -> PROFILED_ctx128k_s42
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue
            out["profiler"][level_id] = data
    # Also pull the per-row profile metrics CSV
    prof_csv = REPO_ROOT / "results/final/profiler_pass/profiler_pass.csv"
    out["profiler"]["_csv_rows"] = _group_by_level(_read_csv(prof_csv),
                                                   group_filter="M4")

    # --- p35c perplexity (both YaRN and vanilla) ---
    for source_csv, key in [
        ("results/perplexity/m4_ppl.csv",              "yarn"),
        ("results/perplexity/m4_ppl_vanilla_rope.csv", "vanilla"),
    ]:
        ppl_rows = _read_csv(REPO_ROOT / source_csv)
        out["perplexity"][key] = []
        for r in ppl_rows:
            out["perplexity"][key].append({
                "seed":           _i(r.get("seed")),
                "context_length": _i(r.get("context_length")),
                "ppl":            _f(r.get("ppl")),
                "quantize_target": r.get("quantize_target") == "True",
                "rope_type":      r.get("rope_type", ""),
                "model":          r.get("model", ""),
            })

    # --- F4 per-position acceptance trace (.jsonl per run, from --log-per-token) ---
    # Each file is JSONL: one record per round with global_pos_start /
    # spec_steps / n_acc / accepted_token_pos / draft_token_pos.
    # Aggregated to: per-run sequence of (position, accept_rate_at_position).
    per_token_dir = REPO_ROOT / "results/final/per_token"
    out["per_position_accept"] = {}
    if per_token_dir.exists():
        for jf in sorted(per_token_dir.glob("*.jsonl")):
            run_id = jf.stem
            records = []
            for line in jf.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if not records:
                continue
            # Reduce to (position, accepted_count, total) per round so F4
            # can plot α(t). The plot script computes rolling means.
            trace = [
                {
                    "round":            r.get("round_idx", r.get("round", 0)),
                    "global_pos_start": r.get("global_pos_start", 0),
                    "spec_steps":       r.get("spec_steps", 0),
                    "n_accepted":       r.get("n_acc", r.get("n_accepted", 0)),
                }
                for r in records
            ]
            out["per_position_accept"][run_id] = trace
    out["_per_position_note"] = (
        "Populated only when per_token/*.jsonl sidecars are present "
        "(produced by run_experiment.py --log-per-token). Phase D F4 "
        "plot script reads from here."
    )

    # --- F5 qualitative text comparison source ---
    # Generated text written by run_experiment.py --save-generated-text.
    # We store paths (not contents) so the JSON stays small.
    gen_dir = REPO_ROOT / "results/final/generated"
    out["generated_text"] = {}
    if gen_dir.exists():
        for tf in sorted(gen_dir.glob("*.txt")):
            run_id = tf.stem
            out["generated_text"][run_id] = {
                "path":   str(tf.relative_to(REPO_ROOT)),
                "bytes":  tf.stat().st_size,
            }

    # --- p35d PG-19 prompt single-cell ---
    pg19_rows = _read_csv(REPO_ROOT / "results/final/p35d_pg19_prompt.csv")
    if pg19_rows:
        r = pg19_rows[0]
        out["ablations"]["pg19_prompt_ctx1M"] = {
            "context_length":  _i(r.get("context_length")),
            "seed":            _i(r.get("seed")),
            "acceptance_rate": _f(r.get("acceptance_rate")),
            "throughput_tps":  _f(r.get("throughput_tps")),
            "gpu_peak_mem_mb": _f(r.get("gpu_peak_mem_mb")),
        }

    # --- p34 baselines (synthetic forwards; flag the units caveat) ---
    base_rows = _read_csv(REPO_ROOT / "results/baselines/m4_baselines.csv")
    base_g: dict = {}
    for r in base_rows:
        b   = r.get("baseline", "")
        ctx = _i(r.get("context_length"))
        tps = _f(r.get("forward_tps"), default=-1.0)
        seed = _i(r.get("seed"))
        if tps < 0:
            continue
        base_g.setdefault(b, []).append({
            "context_length": ctx,
            "seed":           seed,
            "forward_tps":    tps,
        })
    out["baselines"]["_units_caveat"] = (
        "ring/sliding rows are FORWARD-PASS throughput (no decode loop, "
        "no KV cache amortization), NOT generate() tps. Direct y-axis "
        "overlay with RASD's throughput_tps is misleading; use only for "
        "feasibility evidence, not throughput comparison."
    )
    out["baselines"]["ring"]    = base_g.get("ring",    [])
    out["baselines"]["sliding"] = base_g.get("sliding", [])

    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/final/final_results.json")
    p.add_argument("--indent", type=int, default=2)
    args = p.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    with open(out_path, "w") as f:
        json.dump(data, f, indent=args.indent)

    print(f"Wrote {out_path}")
    print(f"  matrix.rasd:        {len(data['matrix']['rasd'])} levels")
    print(f"  matrix.target_only: {len(data['matrix']['target_only'])} levels")
    print(f"  matrix.hf_ceiling:  {len(data['matrix']['hf_ceiling'])} rows")
    print(f"  speedup:            {len(data['speedup'])} contexts")
    print(f"  profiler:           {sum(1 for k in data['profiler'] if not k.startswith('_'))} JSONs")
    print(f"  perplexity.yarn:    {len(data['perplexity']['yarn'])} rows")
    print(f"  perplexity.vanilla: {len(data['perplexity']['vanilla'])} rows")
    print(f"  per_position_accept: {len(data.get('per_position_accept', {}))} runs")
    print(f"  generated_text:      {len(data.get('generated_text', {}))} runs")


if __name__ == "__main__":
    raise SystemExit(main())

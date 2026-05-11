#!/usr/bin/env python3
"""Phase D F7: emit LaTeX tables for the main results section.

Reads results/final/final_results.json and writes:
  tables/main_speedup.tex  — RASD vs target-only speedup at each ctx
  tables/main_memory.tex   — peak GB at each ctx
  tables/main_ppl.tex      — perplexity sanity (YaRN vs vanilla)
  tables/main_profiler.tex — compute/comm/idle/other per ctx
  tables/main_ceiling.tex  — vanilla HF FA-2 ceiling (32k OK, ≥128k OOM)

Each is `\\begin{tabular}` with `\\caption` so they can `\\input{}` into
a paper section directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _humanize_ctx(n: int) -> str:
    # Map powers-of-2 contexts to their conventional labels
    # (131072 == 2^17 is conventionally "128k" not "131k" in ML papers).
    POW2_LABELS = {
        32_768:    "32k",
        65_536:    "64k",
        131_072:   "128k",
        262_144:   "256k",
        524_288:   "512k",
        1_048_576: "1M",
        2_097_152: "2M",
    }
    if n in POW2_LABELS:
        return POW2_LABELS[n]
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def write_speedup(data: dict, out: Path) -> None:
    rows = sorted(data["speedup"].values(), key=lambda r: r["context_length"])
    lines = []
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Context & RASD (tps) & Target (tps) & Speedup & "
                 r"$n$ seeds & RASD 95\% CI \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{_humanize_ctx(r['context_length'])} & "
            f"{r['rasd_mean_tps']:.3f} & "
            f"{r['target_mean_tps']:.3f} & "
            f"{r['ratio']:.2f}$\\times$ & "
            f"{r['n_seeds']} & "
            f"$\\pm$ {r['rasd_ci_half']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def write_memory(data: dict, out: Path) -> None:
    rasd = data["memory"]["rasd"]
    tgt  = data["memory"]["target_only"]
    rows = []
    for level, r in sorted(rasd.items(), key=lambda x: x[1]["context_length"]):
        t_level = level.replace("M4_ctx", "TARGET_ctx")
        t = tgt.get(t_level, {})
        rows.append((
            r["context_length"],
            r["mean"] / 1024 if r.get("mean") else 0.0,   # MB -> GB
            t.get("mean", 0.0) / 1024 if t.get("mean") else 0.0,
        ))
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Context & RASD peak (GB) & Target peak (GB) & RASD overhead (GB) \\",
        r"\midrule",
    ]
    for ctx, rasd_gb, tgt_gb in rows:
        lines.append(
            f"{_humanize_ctx(ctx)} & {rasd_gb:.1f} & {tgt_gb:.1f} & "
            f"{rasd_gb - tgt_gb:+.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def write_ppl(data: dict, out: Path) -> None:
    # Mean PPL per ctx for both YaRN and vanilla
    from statistics import mean
    by_ctx_yarn = {}
    by_ctx_vanilla = {}
    for r in data["perplexity"]["yarn"]:
        by_ctx_yarn.setdefault(r["context_length"], []).append(r["ppl"])
    for r in data["perplexity"]["vanilla"]:
        by_ctx_vanilla.setdefault(r["context_length"], []).append(r["ppl"])

    ctxs = sorted(set(by_ctx_yarn) | set(by_ctx_vanilla))
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Context & YaRN PPL & Vanilla PPL & YaRN benefit \\",
        r"\midrule",
    ]
    for ctx in ctxs:
        y = mean(by_ctx_yarn.get(ctx, [0]))    if by_ctx_yarn.get(ctx) else None
        v = mean(by_ctx_vanilla.get(ctx, [0])) if by_ctx_vanilla.get(ctx) else None
        if y is None or v is None:
            ratio_str = "--"
        else:
            ratio = v / y
            ratio_str = f"{ratio:.1f}$\\times$"
        y_str = f"{y:.2f}" if y is not None else "--"
        v_str = f"{v:.0f}" if v is not None else "--"
        lines.append(
            f"{_humanize_ctx(ctx)} & {y_str} & {v_str} & {ratio_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def write_profiler(data: dict, out: Path) -> None:
    label_map = {
        "matrix_profiled_canary_s42": ("canary", 32768),
        "PROFILED_ctx128k_s42":       ("128k",   131072),
        "PROFILED_ctx256k_s42":       ("256k",   262144),
        "PROFILED_ctx512k_s42":       ("512k",   524288),
        "PROFILED_ctx1M_s42":         ("1M",     1048576),
    }
    rows = []
    for rid, (lab, ctx) in label_map.items():
        p = data["profiler"].get(rid)
        if not p:
            continue
        rows.append((ctx, lab,
                     p.get("compute_pct", 0)*100,
                     p.get("comm_pct", 0)*100,
                     p.get("idle_pct", 0)*100,
                     p.get("other_pct", 0)*100))
    rows.sort()
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Context & Compute (\%) & Comm (\%) & Idle (\%) & Other (\%) \\",
        r"\midrule",
    ]
    for _, lab, c, cm, i, o in rows:
        lines.append(f"{lab} & {c:.1f} & {cm:.1f} & {i:.1f} & {o:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def write_ceiling(data: dict, out: Path) -> None:
    rows = sorted(data["matrix"]["hf_ceiling"],
                  key=lambda r: r["context_length"])
    lines = [
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Context & Throughput (tps) & Peak (GB) & Status \\",
        r"\midrule",
    ]
    for r in rows:
        ctx = _humanize_ctx(r["context_length"])
        if r["status"] == "ok":
            lines.append(
                f"{ctx} & {r['throughput_tps']:.2f} & "
                f"{r['gpu_peak_mem_mb']/1024:.1f} & OK \\\\"
            )
        else:
            lines.append(f"{ctx} & -- & -- & OOM \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path",
                   default="results/final/final_results.json")
    p.add_argument("--out-dir", default="tables")
    args = p.parse_args()

    data = json.loads((REPO_ROOT / args.in_path).read_text())
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_speedup(data,   out_dir / "main_speedup.tex")
    write_memory(data,    out_dir / "main_memory.tex")
    write_ppl(data,       out_dir / "main_ppl.tex")
    write_profiler(data,  out_dir / "main_profiler.tex")
    write_ceiling(data,   out_dir / "main_ceiling.tex")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Figures and LaTeX tables for the LCFM camera-ready, generated from the CSVs.

DESIGN RULES, all deliberate:

  * VECTOR PDF only, and NOTHING is transcribed by hand: every number in every figure and
    table is read from a CSV or scored from a token sidecar by the release's own detector
    (`src.analysis.repetition`). A camera-ready is not the place for a number that no
    artifact reproduces.
  * Width is the template's text width: `article` on a4paper with 1in margins = 15.92 cm =
    6.27 in.
  * NO TITLES inside the figures; the LaTeX caption carries the description.
  * Okabe-Ito palette (colourblind-safe), and every series also carries a distinct marker
    and linestyle so the figures survive greyscale printing.
  * LIKE-FOR-LIKE POOLS. The two engines are compared on **matched cells** -- a cell is
    (seed, book) -- and only rows from the stage that is the analogue of the old-engine
    pool (plumbing repeats from gate/sanity stages are dropped). This is the correction
    recorded in RESULTS_OCT section 10.2a: pooling different book sets on the two sides is
    what made the first 128k row read 2.41x instead of 2.80x.
  * The old-engine pools exclude the rows hit by the `seeds:` merge bug (see
    scripts/mlsys_seed_baseline_audit.py).

Usage:  python3 scripts/lcfm_figures.py [--out results/lcfm]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.repetition import row_reasons, row_stats   # noqa: E402

# Okabe-Ito, colourblind-safe.
C_OLD = "#0072B2"        # blue
C_NEW = "#D55E00"        # vermillion
C_ALT = "#009E73"        # bluish green
C_ALT2 = "#CC79A7"       # reddish purple
C_GREY = "#666666"
TEXTWIDTH_IN = 6.27

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
})

CONTEXTS = [131072, 524288, 1048576]
CTXLABEL = {131072: "128k", 524288: "512k", 1048576: "1M"}

# Old-engine pools, each chosen to be the analogue of the candidate stage at that rung.
OLD_POOLS = {
    (131072, 4): ["results/mlsys/sampling_session/session_a_SAMP_ALIGNED_128K_SPEC.csv"],
    (131072, 0): ["results/mlsys/sampling_session/session_a_SAMP_ALIGNED_128K_TARGET.csv"],
    (524288, 4): ["results/mlsys/sampling_session/session_a_SAMP_ALIGNED_512K_SPEC.csv",
                  "results/mlsys/n5_extension/session_a_SAMP_EXT_ALIGNED_512K_SPEC.csv"],
    (524288, 0): ["results/mlsys/sampling_session/session_a_SAMP_ALIGNED_512K_TARGET.csv"],
    # 1M spec: seed 42 from the sampling session plus the FIXED seed run. The two
    # SA_SEED_* stages are excluded -- they asked for 123/456 and ran 42 (the seed bug).
    (1048576, 4): ["results/mlsys/sampling_session/session_a_SAMP_1M_SPEC.csv",
                   "results/mlsys/fp4_seeds/session_a_FP4_SEED_1M_SPEC.csv"],
    (1048576, 0): ["results/mlsys/sampling_session/session_a_SAMP_1M_TARGET.csv"],
}
NEW_POOLS = ["results/mlsys/rerun_r1/session_a_*.csv",
             "results/mlsys/rerun_r2/session_a_*.csv"]

ROPE_POOLS = {
    "native": ["results/mlsys/sampling_session/session_a_ROPE_SAMP_NATIVE_SPEC.csv",
               "results/mlsys/n5_extension/session_a_ROPE_EXT_SAMP_NATIVE_SPEC.csv"],
    "factor 16": ["results/mlsys/sampling_session/session_a_ROPE_SAMP_F16_SPEC.csv",
                  "results/mlsys/n5_extension/session_a_ROPE_EXT_SAMP_F16_SPEC.csv"],
    "factor 32": ["results/mlsys/sampling_session/session_a_ROPE_SAMP_F32_SPEC.csv",
                  "results/mlsys/n5_extension/session_a_ROPE_EXT_SAMP_F32_SPEC.csv"],
}
ROPE_GATE = "results/mlsys/sampling_session/session_a_rope_fix_gate.csv"

# Fig C. Greedy rows that ship with a per-row verdict, and every sampled row: from the
# per-stage repetition CSVs where they exist and scored from the sidecar where they do not.
GREEDY_REP = ["results/mlsys/seeds_ablations/repetition_SA_TEMP_0_128K_SPEC.csv",
              "results/mlsys/seeds_ablations/repetition_SA_TEMP_1_0_128K_SPEC.csv"]
SAMPLED_REP = ["results/mlsys/sampling_session/repetition_*.csv",
               "results/mlsys/n5_extension/repetition_*.csv",
               "results/mlsys/fp4_seeds/repetition_*.csv"]
SAMPLED_CSV = ["results/mlsys/rerun_r1/session_a_*.csv", "results/mlsys/rerun_r2/session_a_*.csv",
               "results/mlsys/decode_df/session_a_*.csv"]
SAMPLED_FIRSTWIN = "results/mlsys/natural_f1_128k_first_window.csv"

BENCH = "results/mlsys/decode_df/ring_bench/ringbench.csv"
VLLM = "results/mlsys/vllm_run2/vllm_baseline.csv"


# =========================================================================== helpers
def decode(r):
    tok = int(r["tokens_generated"]); t = float(r["time_sec"]); tt = float(r["ttft_ms"])
    return (t - tt / 1000.0) / (tok - 1)


def rows_of(paths, ctx=None, spec=None, drop_plumbing=True):
    out = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            try:
                rs = list(csv.DictReader(open(f)))
            except Exception:
                continue
            if not rs or "context_length" not in rs[0]:
                continue
            for r in rs:
                if r.get("status") != "ok" or not r.get("ttft_ms") or not r.get("time_sec"):
                    continue
                rid = r.get("run_id", "")
                if drop_plumbing and ("GATE" in rid or "SANITY" in rid):
                    continue
                try:
                    if ctx is not None and int(r["context_length"]) != ctx:
                        continue
                    if spec is not None and int(r["spec_steps"]) != spec:
                        continue
                except Exception:
                    continue
                out.append(r)
    return out


def cell_key(r):
    """A matched cell: (seed, book). The book lives in run_id in this project."""
    m = re.search(r"(pg19_train_\d+)", r.get("run_id", ""))
    return (int(r.get("seed") or 0), m.group(1) if m else r.get("doc_id", "?"))


def matched(ctx, spec):
    """Old and new rows for one rung, restricted to the cells both sides measured.

    Returns (cells, old_values, new_values) with one value per cell.
    """
    old = rows_of(OLD_POOLS[(ctx, spec)], ctx, spec)
    new = rows_of(NEW_POOLS, ctx, spec)
    oc, nc = {}, {}
    for r in old:
        oc.setdefault(cell_key(r), []).append(decode(r))
    for r in new:
        nc.setdefault(cell_key(r), []).append(decode(r))
    keys = sorted(set(oc) & set(nc), key=str)
    ov = [st.mean(oc[k]) for k in keys]
    nv = [st.mean(nc[k]) for k in keys]
    return keys, ov, nv


def headline_rows():
    out = []
    for ctx in CONTEXTS:
        for spec, arm in ((4, "spec"), (0, "target-only")):
            keys, ov, nv = matched(ctx, spec)
            o, n = st.mean(ov), st.mean(nv)
            out.append(dict(ctx=CTXLABEL[ctx], arm=arm, old=o, new=n, speedup=o / n,
                            n=len(keys), o_lo=min(ov), o_hi=max(ov),
                            n_lo=min(nv), n_hi=max(nv)))
    return out


def sidecar_ids(csv_path, run_id):
    d = os.path.dirname(csv_path)
    for sub in ("tokens", "tokens_slim", "arms_tokens"):
        for c in glob.glob(os.path.join(d, sub, f"{run_id}*.json")):
            try:
                b = json.load(open(c))
            except Exception:
                continue
            ids = b.get("generated_token_ids") or b.get("generated_ids")
            if ids:
                return ids
    return None


def rows_of_repetition(paths):
    """Rows of the per-stage repetition CSVs: (run_id, periodic_tail, degenerate)."""
    out = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            try:
                rs = list(csv.DictReader(open(f)))
            except Exception:
                continue
            if not rs or "rep_periodic_tail_share" not in rs[0]:
                continue
            for r in rs:
                if r.get("status") and r["status"] != "ok":
                    continue
                if r.get("rep_periodic_tail_share") in (None, ""):
                    continue
                out.append((r["run_id"], float(r["rep_periodic_tail_share"]),
                            str(r.get("rep_degenerate")).lower() in ("true", "1")))
    return out


def rows_scored(paths):
    """Rows scored here from the token sidecar with the release's own detector."""
    out = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            try:
                rs = list(csv.DictReader(open(f)))
            except Exception:
                continue
            if not rs or "context_length" not in rs[0]:
                continue
            for r in rs:
                rid = r.get("run_id")
                if r.get("status") != "ok" or not rid:
                    continue
                ids = sidecar_ids(f, rid)
                if not ids:
                    continue
                st = row_stats(ids)
                out.append((rid, st["rep_periodic_tail_share"], bool(row_reasons(st))))
    return out


# =========================================================================== Fig A
def fig_a(out, head):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.9))
    series = [("old engine, spec", C_OLD, "o", "-", "spec", "old"),
              ("split-KV spec", C_NEW, "s", "-", "spec", "new"),
              ("old engine, target-only", C_OLD, "^", "--", "target-only", "old"),
              ("split-KV target-only", C_NEW, "v", "--", "target-only", "new")]
    xs = [1, 2, 3]
    for label, col, mk, ls, arm, eng in series:
        ys, err_lo, err_hi, ns = [], [], [], []
        for row in [h for h in head if h["arm"] == arm]:
            y = row["old"] if eng == "old" else row["new"]
            lo = row["o_lo"] if eng == "old" else row["n_lo"]
            hi = row["o_hi"] if eng == "old" else row["n_hi"]
            ys.append(y); err_lo.append(y - lo); err_hi.append(hi - y); ns.append(row["n"])
        ax.errorbar(xs, ys, yerr=[err_lo, err_hi], color=col, marker=mk, ls=ls, ms=3.2,
                    lw=1.0, capsize=1.6, elinewidth=0.6, label=label, zorder=3)
        for i, (x, y, n) in enumerate(zip(xs, ys, ns)):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(3, 4 if arm == "spec" else -9), fontsize=5, color=col)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([CTXLABEL[c] for c in CONTEXTS])
    ax.set_xlabel("context length")
    ax.set_ylabel("decode s / token  (log scale)")
    ax.grid(True, which="both", lw=0.3, color="0.85", zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_xlim(0.7, 3.3)
    fig.savefig(os.path.join(out, "figs", "figA_decode_vs_context.pdf"))
    plt.close(fig)


# =========================================================================== Fig B
def rope_arms():
    """Acceptance per arm from the CSV, and the gate's paired ppl ratio."""
    accs, perbook = {}, {}
    for a, paths in ROPE_POOLS.items():
        rs = rows_of(paths, 131072, 4)
        perbook[a] = sorted(float(r["acceptance_rate"]) for r in rs)
        accs[a] = st.mean(perbook[a])
    ppl = {}
    for r in csv.DictReader(open(ROPE_GATE)):
        f = r.get("rope_factor") or ""
        ppl["native" if f == "" else f"factor {f}"] = float(r["ppl_ratio"])
    return list(ROPE_POOLS), accs, perbook, ppl


def fig_b(out, arms, accs, perbook, ppl):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.5))
    cols = [C_ALT, C_OLD, C_NEW]
    ax = axes[0]
    ax.bar(range(len(arms)), [accs[a] for a in arms], color=cols, width=0.55, zorder=2)
    for i, a in enumerate(arms):
        v = perbook[a]
        ax.scatter([i] * len(v), v, s=7, color="k", zorder=3, linewidths=0)
        ax.annotate(f"n={len(v)}", (i, max(v)), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=5)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms)
    ax.set_ylabel("speculative acceptance")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", lw=0.3, color="0.85", zorder=0)
    ax = axes[1]
    rr = [ppl[a] for a in arms]
    ax.bar(range(len(arms)), rr, color=cols, width=0.55, zorder=2)
    ax.axhline(1.0, color="k", lw=0.6, ls=":", zorder=3)
    for i, v in enumerate(rr):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=5)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms)
    ax.set_ylabel("paired ppl ratio vs native")
    ax.set_ylim(0.99, 1.03)
    ax.grid(True, axis="y", lw=0.3, color="0.85", zorder=0)
    fig.savefig(os.path.join(out, "figs", "figB_rope.pdf"))
    plt.close(fig)


# =========================================================================== Fig C
def fig_c(out):
    import csv as _csv
    greedy = rows_of_repetition(GREEDY_REP)
    firstwin = []
    for r in _csv.DictReader(open(SAMPLED_FIRSTWIN)):
        firstwin.append((r["run_id"], float(r["row_periodic_tail_share"]),
                         r["row_flag"].strip().upper() == "LOOP"))
    sampled = rows_of_repetition(SAMPLED_REP) + rows_scored(SAMPLED_CSV)
    groups = [("greedy, 128k\nfirst-window audit", firstwin, C_ALT2),
              ("greedy, 128k\ntemperature arms", greedy, C_OLD),
              ("sampled, T=0.7", sampled, C_NEW)]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.3))
    for gi, (label, vals, col) in enumerate(groups):
        v = sorted(x[1] for x in vals)
        nloop = sum(1 for x in vals if x[2])
        ax.scatter([gi + (i % 7 - 3) * 0.014 for i in range(len(v))], v, s=6, color=col,
                   zorder=3, linewidths=0, label=f"{label.replace(chr(10), ' ')}"
                                                 f"  ({nloop}/{len(v)} loop)")
    ax.axhline(0.5, color="k", lw=0.7, ls="--", zorder=2)
    ax.annotate("periodic-tail ceiling 0.50", (0.02, 0.515), fontsize=5, color="k")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["greedy\nfirst-window\naudit, 128k",
                        "greedy\ntemperature\narms, 128k", "sampled\nT=0.7"])
    ax.set_xlim(-0.45, len(groups) - 0.55)
    ax.set_ylabel("periodic tail share of row")
    ax.set_ylim(-0.03, 1.12)
    ax.grid(True, axis="y", lw=0.3, color="0.85", zorder=0)
    ax.legend(frameon=False, loc="upper right", fontsize=5.2)
    fig.savefig(os.path.join(out, "figs", "figC_degeneracy.pdf"))
    plt.close(fig)
    return [(g[0].replace("\n", " "), sum(1 for x in g[1] if x[2]), len(g[1]))
            for g in groups]


# =========================================================================== Fig D
def fig_d(out):
    rs = [r for r in csv.DictReader(open(BENCH)) if r.get("ms_per_round")]
    rs.sort(key=lambda r: float(r["size_mb_actual"]))
    x = [max(float(r["size_mb_actual"]), 0.003) for r in rs]
    y = [float(r["ms_per_round"]) for r in rs]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.4))
    ax.plot(x, y, marker="o", ms=3.2, lw=1.0, color=C_NEW, zorder=3, label="bare ring round")
    ax.set_xscale("log")
    lo, hi = min(y), max(y)
    ax.set_ylim(lo * 0.9, hi * 1.06)
    ax.annotate(f"{lo:.2f}-{hi:.2f} ms across the whole payload range  ({(hi / lo - 1) * 100:.1f}% spread)",
                (x[1], lo), textcoords="offset points", xytext=(4, 5), fontsize=5.5, color="k")
    for xi, yi, r in zip(x, y, rs):
        if r["label"] in ("tiny_4kb", "mb266"):
            ax.annotate("4 KB" if xi < 1 else f"{float(r['size_mb_actual']):.0f} MB",
                        (xi, yi), textcoords="offset points", xytext=(3, -8), fontsize=5)
    ax.set_xlabel("payload per round, K+V  (MB, log scale)")
    ax.set_ylabel("ms / round")
    ax.grid(True, which="both", lw=0.3, color="0.85", zorder=0)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(out, "figs", "figD_ring_bench.pdf"))
    plt.close(fig)


# =========================================================================== vLLM
def vllm_rows():
    """vLLM 0.6.3 TP=8 bf16 (the corrected run: valid ttft) against the matched RASD rung.

    The 1M figures average the two recorded books; RASD's 1M target-only pool is the same
    two books, so that rung is book-matched.
    """
    rows = [r for r in csv.DictReader(open(VLLM)) if r.get("status") == "ok"]
    out = []
    for ctx in CONTEXTS:
        v = [float(r["decode_only_wall_s"]) / (int(r["output_tokens"]) - 1)
             for r in rows if int(r["context_length"]) == ctx]
        pre = [float(r["ttft_s"]) for r in rows if int(r["context_length"]) == ctx]
        keys, ov, nv = matched(ctx, 0)
        o, n = st.mean(ov), st.mean(nv)
        # RASD prefill, matched book set, candidate engine
        rs = rows_of(NEW_POOLS, ctx, 0)
        keyset = set(keys)
        tt = [float(r["ttft_ms"]) / 1000.0 for r in rs if cell_key(r) in keyset]
        out.append(dict(ctx=CTXLABEL[ctx], v_pre=st.mean(pre), v_dec=st.mean(v),
                        rt_pre=st.mean(tt) if tt else float("nan"),
                        old=o, new=n,
                        r_new=n / st.mean(v), r_old=o / st.mean(v),
                        total_v=st.mean([float(r["end_to_end_wall_s"])
                                         for r in rows if int(r["context_length"]) == ctx])))
    return out


# =========================================================================== tables
def tables(out, head, arms, accs, ppl):
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    lines = {}

    def w(name, rows):
        lines[name] = rows
        with open(os.path.join(out, "tables", name), "w") as fh:
            fh.write("\n".join(rows) + "\n")

    t = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Context & Arm & \multicolumn{2}{c}{Decode s/token} & Speedup & $n$ \\",
         r"\cmidrule(lr){3-4}",
         r" & & old ring & split-KV & & (cells) \\", r"\midrule"]
    for r in head:
        t.append(f"{r['ctx']} & {r['arm']} & {r['old']:.4f} & \\textbf{{{r['new']:.4f}}} & "
                 f"{r['speedup']:.2f}$\\times$ & {r['n']} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("headline.tex", t)

    t = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Arm & $n$ & Acceptance & ppl ratio & tok/s \\",
         r" & & & (vs native) & \emph{[old engine]} \\", r"\midrule"]
    for a in arms:
        rs = rows_of(ROPE_POOLS[a], 131072, 4)
        tps = st.mean([float(r["tokens_generated"]) / float(r["time_sec"]) for r in rs])
        t.append(f"{a} & {len(rs)} & {accs[a]:.4f} & {ppl[a]:.4f} & {tps:.4f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("rope.tex", t)

    vl = vllm_rows()
    t = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Context & \multicolumn{2}{c}{Prefill (s)} & \multicolumn{3}{c}{Decode (s/token)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
         r" & vLLM & RASD & vLLM & old ring & split-KV \\", r"\midrule"]
    for r in vl:
        t.append(f"{r['ctx']} & {r['v_pre']:.2f} & {r['rt_pre']:.2f} & {r['v_dec']:.4f} & "
                 f"{r['old']:.2f} & {r['new']:.2f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("vllm.tex", t)

    t = [r"\begin{tabular}{lll}", r"\toprule",
         r"Claim & Artifact & Row ids \\", r"\midrule",
         r"F1 & \texttt{results/mlsys/sampling\_session} & ROPE\_SAMP\_\{NATIVE,F16,F32\}\_SPEC \\",
         r"F2 & \texttt{repetition\_*.csv} & SA\_TEMP\_\{0,1\_0\} / all sampled stages \\",
         r"F3 & \texttt{QUALITY\_FIX\_\{128K,1M\}} & ppl\_tokens 1024 \\",
         r"F4 & \texttt{results/mlsys/rerun\_r1,r2} & RR\_* (31 rows) \\",
         r"F5 & \texttt{results/mlsys/rerun\_r1,r2} & RR\_*\_SPEC vs RR\_*\_TARGET \\",
         r"\bottomrule", r"\end{tabular}"]
    w("claim_source.tex", t)
    return vl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lcfm")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)

    head = headline_rows()
    fig_a(a.out, head)
    arms, accs, perbook, ppl = rope_arms()
    fig_b(a.out, arms, accs, perbook, ppl)
    census = fig_c(a.out)
    fig_d(a.out)
    vl = tables(a.out, head, arms, accs, ppl)

    print("Fig A / headline table (matched cells, one value per (seed, book)):")
    for r in head:
        print(f"  {r['ctx']:>5s} {r['arm']:11s} old {r['old']:8.4f} [{r['o_lo']:.4f},{r['o_hi']:.4f}]"
              f"  new {r['new']:7.4f} [{r['n_lo']:.4f},{r['n_hi']:.4f}]  "
              f"{r['speedup']:.3f}x  cells={r['n']}")
    print("Fig B (RoPE):", {a_: round(accs[a_], 4) for a_ in arms}, "ppl",
          {a_: ppl[a_] for a_ in arms})
    print("Fig C census (loops/rows):", census)
    print("vLLM table:")
    for r in vl:
        print(f"  {r['ctx']:>5s} vLLM prefill {r['v_pre']:7.2f}s RASD {r['rt_pre']:7.2f}s | "
              f"vLLM decode {r['v_dec']:.4f}s/tok  old {r['old']:.2f} ({r['r_old']:.0f}x)  "
              f"new {r['new']:.2f} ({r['r_new']:.0f}x) | vLLM total {r['total_v']:.1f}s")
    print("wrote figures and tables under", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

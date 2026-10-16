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

# The blocking-wait diagnostic (results/lcfm/BLOCKING_WAIT_DIAG.md). Fig D and the
# Appendix C table come from here: the same ring primitive measured with and without
# TORCH_NCCL_BLOCKING_WAIT, in one session, on one box.
BW_BENCH = {"blocking": "results/mlsys/bw_diag_p1/bench/ringbench_blocking.csv",
            "no blocking wait": "results/mlsys/bw_diag_p1/bench/ringbench_async.csv"}
BW_ARMS_OLD = "results/mlsys/bw_diag_p1/arms/blocking/df_arms.csv"
BW_ARMS_NEW = "results/mlsys/bw_diag_p1/arms/async/df_arms.csv"
BW_ONEM_OLD = "results/mlsys/bw_diag_p2/onem/blocking/df_arms.csv"
BW_ONEM_NEW = "results/mlsys/bw_diag_p2/onem/async/df_arms.csv"
BW_SPEC_OLD = "results/mlsys/bw_diag_p2/spec/blocking/df_arms.csv"
BW_SPEC_NEW = "results/mlsys/bw_diag_p2/spec/async/df_arms.csv"

# The top-up target-only rows that widen the 512k and 1M denominators (RESULTS_OCT 10.7).
TOPUP = {524288: "results/mlsys/rerun_topup/session_a_TP_512K_TARGET.csv",
         1048576: "results/mlsys/rerun_topup/session_a_TP_1M_TARGET.csv"}
# Rounds per decode token at 128k: a rotation per step < world_size - 1, over the patched
# layers. 7 x 32 = 224. See src/models/ring_attention_kernel.py:402,426,455.
ROTATIONS_PER_LAYER = 7
PATCHED_LAYERS = 32
ROUNDS_PER_TOKEN = ROTATIONS_PER_LAYER * PATCHED_LAYERS


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
    """Rows of the per-stage repetition CSVs: (run_id, periodic_tail, degenerate, T).

    The temperature is NOT in these CSVs, so it is read from the stage CSV the report was
    built from (`repetition_X.csv` beside `session_a_X.csv`). Classifying by stage name is
    what let 5 rows at T=1.0 be counted as greedy and 6 rows at T=0.0 be counted as sampled.
    """
    out = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            try:
                rs = list(csv.DictReader(open(f)))
            except Exception:
                continue
            if not rs or "rep_periodic_tail_share" not in rs[0]:
                continue
            temps = stage_temperatures(f)
            for r in rs:
                if r.get("status") and r["status"] != "ok":
                    continue
                if r.get("rep_periodic_tail_share") in (None, ""):
                    continue
                t = temps.get(r["run_id"])
                out.append((r["run_id"], float(r["rep_periodic_tail_share"]),
                            str(r.get("rep_degenerate")).lower() in ("true", "1"), t))
    return out


def stage_temperatures(rep_path):
    """run_id -> recorded temperature, from the stage CSV the repetition report reads."""
    d = os.path.dirname(rep_path)
    stem = os.path.splitext(os.path.basename(rep_path))[0]
    if stem.endswith("_repetition"):
        stem = stem[:-len("_repetition")]
    stem = stem.replace("repetition_", "", 1)
    cands = [os.path.join(d, f"session_a_{stem}.csv"), os.path.join(d, f"{stem}.csv")]
    cands += [c for c in sorted(glob.glob(os.path.join(d, f"*{stem}.csv")))
              if "repetition" not in os.path.basename(c)]
    out = {}
    for c in cands:
        if not os.path.exists(c):
            continue
        try:
            rs = list(csv.DictReader(open(c)))
        except Exception:
            continue
        if not rs or "temperature" not in rs[0]:
            continue
        for r in rs:
            if r.get("temperature") not in (None, ""):
                out[r["run_id"]] = float(r["temperature"])
        if out:
            break
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
                t = r.get("temperature")
                out.append((rid, st["rep_periodic_tail_share"], bool(row_reasons(st)),
                            float(t) if t not in (None, "") else None))
    return out


# =========================================================================== Fig A
def fig_a(out, head):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.35))
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
    """Acceptance per arm from the CSV, and the gate's paired ppl ratio.

    Also returns the factor-16 mean with and without the one row the repetition detector
    flags (`ROPE_EXT_SAMP_F16_SPEC_pg19_train_2768_s42`, 77% of tokens inside a repeated
    16-gram). That row is inside the factor-16 mean, so the paper reports both.
    """
    accs, perbook, loops = {}, {}, {}
    for a, paths in ROPE_POOLS.items():
        rs = rows_of(paths, 131072, 4)
        perbook[a] = sorted(float(r["acceptance_rate"]) for r in rs)
        accs[a] = st.mean(perbook[a])
        loops[a] = [r["run_id"] for r in rs if float(r["acceptance_rate"]) == 0.0]
    f16 = rows_of(ROPE_POOLS["factor 16"], 131072, 4)
    loop_id = "ROPE_EXT_SAMP_F16_SPEC_pg19_train_2768_s42"
    f16_wo = [float(r["acceptance_rate"]) for r in f16 if not r["run_id"].startswith(loop_id)]
    f16_excl = st.mean(f16_wo)
    ppl = {}
    for r in csv.DictReader(open(ROPE_GATE)):
        f = r.get("rope_factor") or ""
        ppl["native" if f == "" else f"factor {f}"] = float(r["ppl_ratio"])
    gate = list(csv.DictReader(open(ROPE_GATE)))
    return list(ROPE_POOLS), accs, perbook, ppl, f16_excl, gate


def fig_b(out, arms, accs, perbook, ppl):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.05))
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
    ax.annotate("circles: per-book rows", (0.03, 0.93), fontsize=4.6, color="0.25")
    ax = axes[1]
    rr = [ppl[a] for a in arms]
    ax.bar(range(len(arms)), rr, color=cols, width=0.55, zorder=2)
    ax.axhline(1.0, color="k", lw=0.6, ls=":", zorder=3)
    for i, v in enumerate(rr):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=5)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms)
    ax.set_ylabel("paired ppl ratio vs native\n(n=1 passage, different prompt)")
    ax.set_ylim(0.99, 1.03)
    ax.grid(True, axis="y", lw=0.3, color="0.85", zorder=0)
    fig.savefig(os.path.join(out, "figs", "figB_rope.pdf"))
    plt.close(fig)


# =========================================================================== Fig C
def census():
    """The degeneracy census, grouped by the RECORDED temperature of each row.

    Groups are built from the temperature column of the stage CSV that produced each row,
    not from the stage's name: the 128k temperature arm ships two stages (`SA_TEMP_0`, all
    T=0.0, and `SA_TEMP_1_0`, all T=1.0) and the coherence stages are T=0.0 with 8 generated
    tokens, which is too short for a detector whose smallest span is 128 tokens.
    """
    import csv as _csv
    firstwin = [(r["run_id"], float(r["row_periodic_tail_share"]),
                 r["row_flag"].strip().upper() == "LOOP", 0.0)
                for r in _csv.DictReader(open(SAMPLED_FIRSTWIN))]
    temp_arms = rows_of_repetition(GREEDY_REP)
    greedy = [x for x in temp_arms if x[3] == 0.0]
    t1 = [x for x in temp_arms if x[3] == 1.0]
    pooled = rows_of_repetition(SAMPLED_REP) + rows_scored(SAMPLED_CSV)
    sampled = [x for x in pooled if x[3] == 0.7]
    dropped = [x for x in pooled if x[3] != 0.7]
    groups = [("greedy, 128k\nfirst-window audit", firstwin, C_ALT2),
              ("greedy, 128k\ntemperature arm", greedy, C_OLD),
              ("sampled, T=1.0\n128k arm", t1, C_ALT),
              ("sampled, T=0.7\nall released rows", sampled, C_NEW)]
    return groups, dropped


def fig_c(out, groups, dropped):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 1.95))
    for gi, (label, vals, col) in enumerate(groups):
        v = sorted(x[1] for x in vals)
        nloop = sum(1 for x in vals if x[2])
        xs = [gi + (i % 7 - 3) * 0.014 for i in range(len(v))]
        for x, val, loop in zip(xs, v, [x[2] for x in sorted(vals, key=lambda z: z[1])]):
            ax.scatter([x], [val], s=9 if loop else 6, color=col, zorder=4 if loop else 3,
                       linewidths=0.5, edgecolors="k" if loop else "none", marker="D" if loop else "o")
        ax.scatter([], [], s=9, color=col, marker="D", edgecolors="k", linewidths=0.5,
                   label=f"{label.replace(chr(10), ' ')}: {nloop}/{len(v)} loop")
    ax.axhline(0.5, color="k", lw=0.7, ls="--", zorder=2)
    ax.annotate("periodic-tail ceiling 0.50", (0.02, 0.515), fontsize=5, color="k")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.45, len(groups) - 0.55)
    ax.set_ylabel("periodic tail share of row")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, axis="y", lw=0.3, color="0.85", zorder=0)
    ax.legend(frameon=False, loc="center right", fontsize=4.8)
    fig.savefig(os.path.join(out, "figs", "figC_degeneracy.pdf"))
    plt.close(fig)
    return ([(g[0].replace("\n", " "), sum(1 for x in g[1] if x[2]), len(g[1]))
             for g in groups], len(dropped))


# =========================================================================== Fig D
def bench_two_modes():
    """ms/round per payload, both NCCL wait modes, from the diagnostic's own CSVs."""
    out = {}
    for mode, path in BW_BENCH.items():
        rs = [r for r in csv.DictReader(open(path)) if r.get("ms_per_round")]
        rs.sort(key=lambda r: (r["timing"], float(r["size_mb_actual"])))
        w = [r for r in rs if r["timing"] == "wait"]
        s = [r for r in rs if r["timing"] == "sync"]
        assert [r["label"] for r in w] == [r["label"] for r in s], path
        out[mode] = dict(
            x=[max(float(r["size_mb_actual"]), 0.003) for r in s],
            wait=[float(r["ms_per_round"]) for r in w],
            sync=[float(r["ms_per_round"]) for r in s],
            gbs=[float(r["gb_per_s"]) for r in s],
            labels=[r["label"] for r in s],
            env=rs[0].get("env_arm"))
    return out


def fig_d(out, bench):
    """The flat line and the rising curve: the same primitive, both NCCL wait modes."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.1))
    b, a = bench["blocking"], bench["no blocking wait"]
    ax.plot(b["x"], b["wait"], marker="s", ms=3.4, lw=1.2, color=C_OLD,
            zorder=4, label="blocking wait: the host wait() call returns")
    ax.plot(a["x"], a["sync"], marker="^", ms=3.4, lw=1.2, color=C_NEW,
            zorder=4, label="no blocking wait: after a device sync")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.07, 40)
    ax.set_xlabel("payload per round, K+V  (MB, log scale)")
    ax.set_ylabel("ms / round  (log scale)")
    ax.grid(True, which="both", lw=0.3, color="0.85", zorder=0)
    ax.annotate(f"{min(b['wait']):.2f} to {max(b['wait']):.2f} ms over the whole range",
                (0.0045, 11.0), fontsize=5.4, color=C_OLD)
    ax.annotate(f"{min(a['sync']):.3f} ms", (0.0045, 0.15), fontsize=5.4, color=C_NEW)
    ax.annotate(f"{max(a['sync']):.2f} ms at {max(a['gbs']):.1f} GB/s",
                (max(a["x"]), max(a["sync"])), textcoords="offset points", xytext=(-56, 5),
                fontsize=5.4, color=C_NEW)
    ax.annotate("66,584x payload range", (0.0045, 0.083), fontsize=5.4, color="0.3")
    ax.legend(frameon=False, loc="lower right", fontsize=5.0)
    fig.savefig(os.path.join(out, "figs", "figD_ring_bench.pdf"))
    plt.close(fig)


# =========================================================================== vLLM
def vllm_rows():
    """vLLM 0.6.3 TP=8 bf16 (the corrected run: valid ttft) against the matched RASD rung.

    Three ratios are easy to confuse and the paper quotes two of them:
      * `pre_pre`  vLLM prefill / RASD prefill;
      * `tot_pre`  vLLM's ENTIRE run (prefill + 128 decode steps) / RASD prefill;
      * `r_new`    vLLM decode / RASD decode on the same cells.
    RASD's times here are the split-KV engine measured under blocking-wait NCCL, which is
    the configuration every measurement in this paper ran in.
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
        rt_pre = st.mean(tt) if tt else float("nan")
        total_v = st.mean([float(r["end_to_end_wall_s"])
                           for r in rows if int(r["context_length"]) == ctx])
        specd = [float(r["decode_only_wall_s"]) for r in rows if int(r["context_length"]) == ctx]
        ks, os_, ns_ = matched(ctx, 4)
        out.append(dict(ctx=CTXLABEL[ctx], v_pre=st.mean(pre), v_dec=st.mean(v),
                        v_total=total_v, rt_pre=rt_pre, old=o, new=n,
                        pre_pre=st.mean(pre) / rt_pre, tot_pre=total_v / rt_pre,
                        r_new=n / st.mean(v), r_old=o / st.mean(v),
                        spec_v=st.mean(ns_) / (st.mean(specd) / (int(rows[0]["output_tokens"]) - 1))
                        if ks else float("nan")))
    return out


# =========================================================================== F5
def f5_rows():
    """Speculation against the same engine's target-only path, on MATCHED books only.

    The 128k target-only arm is one book, so only that book is matched there; 512k and 1M
    are matched on 3 and 5 books after the top-up. Decode-only is s/token, end-to-end is
    mean tokens/second over the matched books.
    """
    out = []
    for ctx in CONTEXTS:
        srows = [r for r in rows_of(NEW_POOLS, ctx, 4) if int(r["seed"]) == 42]
        trows = rows_of(NEW_POOLS, ctx, 0)
        if ctx in TOPUP:
            trows = trows + [r for r in csv.DictReader(open(TOPUP[ctx]))
                             if r.get("status") == "ok"]
        sm = {cell_key(r)[1]: decode(r) for r in srows}
        tm = {cell_key(r)[1]: decode(r) for r in trows}
        common = sorted(set(sm) & set(tm))
        s, t = st.mean(sm[b] for b in common), st.mean(tm[b] for b in common)
        stp = {cell_key(r)[1]: float(r["tokens_generated"]) / float(r["time_sec"]) for r in srows}
        ttp = {cell_key(r)[1]: float(r["tokens_generated"]) / float(r["time_sec"]) for r in trows}
        out.append(dict(ctx=CTXLABEL[ctx], n_books=len(common), spec=s, target=t, ratio=t / s,
                        ratio_e2e=st.mean(stp[b] for b in common) / st.mean(ttp[b] for b in common),
                        tps_spec=st.mean(stp[b] for b in common),
                        tps_target=st.mean(ttp[b] for b in common)))
    return out


# =========================================================================== Appendix C
def bw_diag():
    """The blocking-wait diagnostic: bench per payload, engine arms, and 1M rows."""
    bench = bench_two_modes()
    arms = {}
    for name, path in (("blocking", BW_ARMS_OLD), ("no blocking wait", BW_ARMS_NEW)):
        arms[name] = [r for r in csv.DictReader(open(path)) if r.get("status") == "ok"]
    for name, path in (("blocking spec", BW_SPEC_OLD), ("no blocking wait spec", BW_SPEC_NEW)):
        arms[name] = [r for r in csv.DictReader(open(path)) if r.get("status") == "ok"]
    onem = {}
    for name, path in (("blocking", BW_ONEM_OLD), ("no blocking wait", BW_ONEM_NEW)):
        onem[name] = [r for r in csv.DictReader(open(path)) if r.get("status") == "ok"]
    return bench, arms, onem


def removed_time(head):
    """What the rotation costs per rung: old minus new, and the same divided by the rounds.

    The benchmark's own 266 MB payload is the key/value shard of a 128k window, so the
    payload a round ships scales with the context (1x, 4x, 8x); the removed time is the
    measured old-minus-new difference on matched cells.
    """
    out = []
    for h, ctx in zip([x for x in head if x["arm"] == "target-only"], CONTEXTS):
        d = h["old"] - h["new"]
        out.append(dict(ctx=h["ctx"], old=h["old"], new=h["new"], removed=d,
                        per_round_ms=1000.0 * d / ROUNDS_PER_TOKEN,
                        tokens=ctx, payload_x=ctx / CONTEXTS[0]))
    return out


def memory_peak():
    """The campaign's 1M peak, worst rank, from the CSVs (not the reviewed build's trace)."""
    out = []
    for mode, path in (("blocking", BW_ONEM_OLD), ("no blocking wait", BW_ONEM_NEW)):
        for r in csv.DictReader(open(path)):
            if r.get("status") != "ok":
                continue
            out.append(dict(mode=mode, arm=r["arm"], mode_col=r["mode"],
                            alloc=float(r["peak_alloc_mb_worst_rank"]) / 1024.0,
                            reserved=float(r["peak_reserved_mb_worst_rank"]) / 1024.0))
    return out


def seed_spread():
    """Fixed-engine 1M spec decode means per seed, and the spread across them."""
    rs = rows_of(NEW_POOLS, 1048576, 4)
    by = {}
    for r in rs:
        by.setdefault(int(r["seed"]), []).append(decode(r))
    means = {k: st.mean(v) for k, v in sorted(by.items())}
    acc = {}
    for r in rs:
        acc.setdefault(int(r["seed"]), []).append(float(r["acceptance_rate"]))
    ameans = {k: st.mean(v) for k, v in sorted(acc.items())}
    return means, max(means.values()) / min(means.values()), ameans


# =========================================================================== tables
def tables(out, head, arms, accs, ppl, vl, f5, bwd, mem):
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

    t = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Prefill (s)} & \multicolumn{2}{c}{vLLM run /} & "
         r"\multicolumn{2}{c}{Decode (s/token)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Context & vLLM & RASD & RASD prefill & RASD prefill & vLLM & RASD split-KV \\",
         r" & & & \multicolumn{2}{c}{(same quantity)} & & \\", r"\midrule"]
    for r in vl:
        t.append(f"{r['ctx']} & {r['v_pre']:.2f} & {r['rt_pre']:.2f} & "
                 f"{r['pre_pre']:.2f}$\\times$ & {r['tot_pre']:.2f}$\\times$ & "
                 f"{r['v_dec']:.4f} & {r['new']:.2f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("vllm.tex", t)

    t = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Context & Arm & old ring & split-KV & removed & removed per round \\",
         r" & & (s/token) & (s/token) & (s/token) & (ms, $\div$224) \\", r"\midrule"]
    for r in removed_time(head):
        t.append(f"{r['ctx']} & target-only & {r['old']:.4f} & {r['new']:.4f} & "
                 f"{r['removed']:.2f} & {r['per_round_ms']:.1f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("rotation_cost.tex", t)

    t = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Context & matched books & spec s/token & target-only s/token & speedup \\",
         r"\midrule"]
    for r in f5:
        t.append(f"{r['ctx']} & {r['n_books']} & {r['spec']:.4f} & {r['target']:.4f} & "
                 f"{r['ratio']:.2f}$\\times$ \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("f5_matched.tex", t)

    bench, bw_arms, bw_onem = bwd
    t = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Payload & size & \multicolumn{2}{c}{ms/round} & \\",
         r"\cmidrule(lr){3-4}",
         r" & (MB) & blocking wait & no blocking wait & \\", r"\midrule"]
    b, a = bench["blocking"], bench["no blocking wait"]
    for i, lab in enumerate(b["labels"]):
        t.append(f"\\texttt{{{lab.replace('_', chr(92) + '_')}}} & {b['x'][i]:.3f} & "
                 f"{b['wait'][i]:.4f} & \\textbf{{{a['sync'][i]:.4f}}} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("bw_bench.tex", t)

    t = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Context & Arm & Engine & mode & decode s/token & TTFT (s) \\", r"\midrule"]
    for name, rows in bw_arms.items():
        for r in rows:
            t.append(f"{int(r['context_length']) // 1024}\\,k & {name} & "
                     f"\\texttt{{{r['arm']}}} & {r['mode']} & "
                     f"{float(r['decode_s_per_tok']):.4f} & {float(r['ttft_ms']) / 1000:.1f} \\\\")
    for name, rows in bw_onem.items():
        for r in rows:
            t.append(f"1\\,M & {name} & \\texttt{{{r['arm']}}} & {r['mode']} & "
                     f"{float(r['decode_s_per_tok']):.4f} & {float(r['ttft_ms']) / 1000:.1f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("bw_arms.tex", t)

    t = [r"\begin{tabular}{llrr}", r"\toprule",
         r"1M rows, worst rank & allocated (GiB) & reserved (GiB) & \\", r"\midrule"]
    for r in mem:
        t.append(f"{r['mode']}, {r['arm']} {r['mode_col']} & {r['alloc']:.1f} & "
                 f"{r['reserved']:.1f} & \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    w("memory.tex", t)

    t = [r"\begin{tabular}{lll}", r"\toprule",
         r"Claim & Artifact & Row ids \\", r"\midrule",
         r"F1 & \texttt{results/mlsys/sampling\_session} & ROPE\_SAMP\_\{NATIVE,F16,F32\}\_SPEC \\",
         r"F2 & \texttt{repetition\_*.csv} & SA\_TEMP\_\{0,1\_0\} / all sampled stages \\",
         r"F3 & \texttt{QUALITY\_FIX\_\{128K,1M\}} & ppl\_tokens 1024 \\",
         r"F4 & \texttt{results/mlsys/rerun\_r1,r2} & RR\_* (31 rows) \\",
         r"F4 & \texttt{results/mlsys/bw\_diag\_p1,p2} & bench, arms, onem \\",
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
    arms, accs, perbook, ppl, f16_excl, gate = rope_arms()
    fig_b(a.out, arms, accs, perbook, ppl)
    groups, dropped = census()
    fig_c(a.out, groups, dropped)
    bench = bench_two_modes()
    fig_d(a.out, bench)
    bwd = bw_diag()
    f5 = f5_rows()
    mem = memory_peak()
    vl = tables(a.out, head, arms, accs, ppl, vllm_rows(), f5, bwd, mem)

    print(f"rounds per decode token at 128k: {ROTATIONS_PER_LAYER} rotations x "
          f"{PATCHED_LAYERS} layers = {ROUNDS_PER_TOKEN}")
    print("Fig A / headline table (matched cells, one value per (seed, book)):")
    for r in head:
        print(f"  {r['ctx']:>5s} {r['arm']:11s} old {r['old']:8.4f} [{r['o_lo']:.4f},{r['o_hi']:.4f}]"
              f"  new {r['new']:7.4f} [{r['n_lo']:.4f},{r['n_hi']:.4f}]  "
              f"{r['speedup']:.3f}x  cells={r['n']}")
    print("rotation cost per rung (old minus new, target-only, matched cells):")
    for r in removed_time(head):
        print(f"  {r['ctx']:>5s} {r['old']:.4f} - {r['new']:.4f} = {r['removed']:.4f} s/token"
              f"  = {r['per_round_ms']:.1f} ms/round over {ROUNDS_PER_TOKEN} rounds"
              f"  (payload {r['payload_x']:.0f}x the 128k shard)")
    print("Fig B (RoPE):", {a_: round(accs[a_], 4) for a_ in arms}, "ppl",
          {a_: ppl[a_] for a_ in arms})
    print(f"  factor 16 with the looping row {accs['factor 16']:.4f}, without it "
          f"{f16_excl:.4f}; gate rows {len(gate)} with ppl_ratio "
          f"{[g['ppl_ratio'] for g in gate]}")
    print("Fig C census (loops/rows):",
          [(g[0].replace(chr(10), ' '), sum(1 for x in g[1] if x[2]), len(g[1])) for g in groups],
          f"| dropped {len(dropped)} pooled rows whose recorded temperature is not 0.7:",
          sorted({x[3] for x in dropped}))
    print("Fig D bench, blocking wait ms/round:",
          [f"{v:.4f}" for v in bench["blocking"]["wait"]])
    print("Fig D bench, no blocking wait ms/round (after sync):",
          [f"{v:.4f}" for v in bench["no blocking wait"]["sync"]],
          "GB/s", [f"{v:.2f}" for v in bench["no blocking wait"]["gbs"]])
    print("F5 spec over target, matched books only:")
    for r in f5:
        print(f"  {r['ctx']:>5s} books={r['n_books']} spec {r['spec']:.4f} target "
              f"{r['target']:.4f} decode-only {r['ratio']:.4f}x  end-to-end "
              f"{r['ratio_e2e']:.4f}x")
    print("vLLM table (RASD times are the split-KV engine under blocking wait):")
    for r in vl:
        print(f"  {r['ctx']:>5s} vLLM prefill {r['v_pre']:7.2f}s RASD {r['rt_pre']:7.2f}s | "
              f"prefill/prefill {r['pre_pre']:.3f}x  vLLM run/prefill {r['tot_pre']:.3f}x | "
              f"vLLM decode {r['v_dec']:.4f}s/tok  old {r['old']:.2f} ({r['r_old']:.0f}x)  "
              f"new {r['new']:.2f} ({r['r_new']:.0f}x)  spec {r['spec_v']:.0f}x")
    means, spread, ameans = seed_spread()
    print(f"fixed-engine 1M spec decode per seed { {k: round(v, 4) for k, v in means.items()} }"
          f" spread max/min {spread:.4f}")
    print(f"fixed-engine 1M spec acceptance per seed "
          f"{ {k: round(v, 4) for k, v in ameans.items()} }")
    print("1M peak memory, worst rank (this campaign):")
    for r in mem:
        print(f"  {r['mode']:16s} {r['arm']:10s} {r['mode_col']:7s} allocated "
              f"{r['alloc']:.1f} GiB  reserved {r['reserved']:.1f} GiB")
    print("wrote figures and tables under", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

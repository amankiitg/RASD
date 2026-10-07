#!/usr/bin/env python3
"""MLSys experiment program — master old-vs-new table.

Emits results/mlsys/MASTER_TABLE.txt: one row per reviewer / roadmap
item, with the previously-reported number and the value the MLSys run
produces. Re-run after the pod session to fill in the new column.

Design rules (so the table cannot overstate what exists):

  * "old" is computed from the CSVs already in the repo, with the file
    named inline. It is never typed in by hand.
  * "new" is read from results/mlsys/. If the pod has not produced the
    cell yet it prints "(pending GPU run)" — it never falls back to the
    old value, and never interpolates or estimates.
  * every row states the seed count it is based on, so a 1-seed number
    cannot be mistaken for a 3-seed one.

Usage:
    python scripts/mlsys_master_table.py
    python scripts/mlsys_master_table.py --out results/mlsys/MASTER_TABLE.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

PENDING = "(pending GPU run)"


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _alpha(df: pd.DataFrame | None, **filters) -> tuple[str, int]:
    """Mean acceptance_rate over rows matching filters -> ('0.0903 (n=3)', 3)."""
    if df is None or df.empty:
        return PENDING, 0
    sub = df
    for col, val in filters.items():
        if col in sub.columns:
            sub = sub[sub[col] == val]
    vals = pd.to_numeric(sub.get("acceptance_rate"), errors="coerce").dropna()
    if vals.empty:
        return PENDING, 0
    return f"{vals.mean():.4f} (n={len(vals)})", len(vals)


def _cell(df: pd.DataFrame | None, col: str, **filters) -> str:
    if df is None or df.empty or col not in df.columns:
        return PENDING
    sub = df
    for k, v in filters.items():
        if k in sub.columns:
            sub = sub[sub[k] == v]
    vals = pd.to_numeric(sub[col], errors="coerce").dropna()
    if vals.empty:
        return PENDING
    return f"{vals.mean():.4f} (n={len(vals)})"


def build_rows() -> list[dict]:
    fm = _read(REPO / "results/final/final_matrix.csv")
    p35d = _read(REPO / "results/final/p35d_pg19_prompt.csv")
    phd = _read(REPO / "results/final/phase_d_short_ctx_pg19.csv")

    arm1 = _read(REPO / "results/mlsys/arm1_llama2_yarn.csv")
    arm2 = _read(REPO / "results/mlsys/arm2_native_cappeddraft.csv")
    arm3 = _read(REPO / "results/mlsys/arm3_native_nativedraft.csv")
    pg19 = _read(REPO / "results/mlsys/pg19_multiseed.csv")
    bftd = _read(REPO / "results/mlsys/bf16_draft_isolation.csv")
    dip = _read(REPO / "results/mlsys/dip_by_context.csv")
    vllm = _read(REPO / "results/mlsys/vllm_baseline.csv")

    rows: list[dict] = []

    # ---- Phase 1: the headline -----------------------------------------
    o1, _ = _alpha(fm, level_id="M4_ctx128k")
    rows.append({
        "item": "Phase 1 / reviewer #1 (native vs YaRN)",
        "question": "Is the 128k acceptance drop caused by YaRN/OOD RoPE, "
                    "or by context length itself?",
        "old": f"Llama-2-7B+YaRN @128k alpha={o1} "
               "[results/final/final_matrix.csv]",
        "new": f"Arm1 (YaRN)={_alpha(arm1, group='ARM1_SPEC')[0]}  "
               f"Arm2 (Llama-3.1 native, draft capped 4k)="
               f"{_alpha(arm2, group='ARM2_SPEC')[0]}  "
               f"Arm3 (native draft window)="
               f"{_alpha(arm3, group='ARM3_SPEC')[0]}",
        "read": "Arm1 vs Arm2: if Arm2 recovers -> YaRN/OOD was the driver. "
                "Arm2 vs Arm3: the draft-window effect, isolated.",
    })

    # ---- Phase 2a: PG-19 dose-response ---------------------------------
    o2, n2 = _alpha(phd, level_id="RASD_ctx4k_pg19_phaseD")
    o2b, _ = _alpha(phd, level_id="RASD_ctx8k_pg19_phaseD")
    o2c, _ = _alpha(p35d, level_id="P35D_ctx1M_pg19")
    rows.append({
        "item": "Phase 2.1 / 'largely single-seed'",
        "question": "PG-19 dose-response (4k/8k/1M) with 3-seed CIs",
        "old": f"4k={o2}, 8k={o2b}, 1M={o2c} — ALL SINGLE-SEED (seed 42) "
               "[results/final/phase_d_short_ctx_pg19.csv, p35d_pg19_prompt.csv]",
        "new": f"4k={_alpha(pg19, level_id='PG19_ctx4k')[0]}  "
               f"8k={_alpha(pg19, level_id='PG19_ctx8k')[0]}  "
               f"1M={_alpha(pg19, level_id='PG19_ctx1M')[0]}  "
               f"(seeds 123,456 added to the existing 42)",
        "read": "3-seed CI on every dose-response point.",
    })
    rows.append({
        "item": "Phase 2.1b / missing paired baseline",
        "question": "1M PG-19 had no target-only denominator",
        "old": "no 1M PG-19 target-only row exists (speedup undefined)",
        "new": f"1M target-only alpha="
               f"{_alpha(pg19, level_id='PG19_ctx1M_targetonly')[0]}",
        "read": "Gives the 1M PG-19 cell a matched denominator.",
    })

    # ---- Phase 2b: per-round traces ------------------------------------
    rows.append({
        "item": "Phase 2.2 / bimodality needs per-round data",
        "question": "128k/256k/512k/1M per-round traces for 3 seeds",
        "old": "per-round traces exist for seed 42 only "
               "[results/final/per_token/RASD_ctx*_phaseD_s42.jsonl]",
        "new": "seeds 123/456 traces -> results/mlsys/per_token/ "
               f"({_count(REPO / 'results/mlsys/per_token', 'jsonl')} files present)",
        "read": "Feeds the Phase 4 dip test and the P(alpha=0) CIs.",
    })

    # ---- Phase 3: mechanism isolation ----------------------------------
    rows.append({
        "item": "Phase 3 / roadmap T3.3 (round-cost source not isolated)",
        "question": "Does 4-bit-FP4 draft/target logit divergence drive the "
                    "round-cost increase, rather than context length?",
        "old": "no bf16-draft long-context comparison existed "
               "(only scripts/run_quant_ablation.py at short context)",
        "new": f"64k FP4 draft alpha="
               f"{_alpha(bftd, level_id='BFTD_ctx64k_draftnf4')[0]}  "
               f"64k bf16 draft alpha="
               f"{_alpha(bftd, level_id='BFTD_ctx64k_draftbf16')[0]}",
        "read": "If bf16-draft acceptance is materially higher at equal "
                "context, NF4 divergence is a driver.",
    })

    # ---- Phase 4: statistics -------------------------------------------
    if dip is not None and not dip.empty:
        # Render per (family, context) so the native-vs-YaRN arms and the
        # M4 dose-response never read as one pooled series at shared
        # contexts (e.g. both have a 128k cell).
        parts = []
        by_fam: dict = {}
        for _, r in dip.iterrows():
            ctx = int(r["context_length"]) if pd.notna(r["context_length"]) else "?"
            # Report rejecting SEEDS (the unit a reader cares about) and
            # rejecting RUNS when they differ, so a context with several
            # variants per seed cannot read as "5/3 seeds".
            n_seed_rej = int(r["n_seeds_reject"])
            n_seeds = int(r["n_seeds"])
            txt = f"{n_seed_rej}/{n_seeds} seeds reject"
            if "n_runs_reject" in r and int(r["n_runs_reject"]) != n_seed_rej:
                txt += (f" ({int(r['n_runs_reject'])}/{int(r['n_runs'])} "
                        f"runs reject)")
            fam = str(r.get("family", "matrix"))
            by_fam.setdefault(fam, []).append(
                f"{ctx}: dip={r['dip_mean']:.3f}, {txt}")
        for fam in sorted(by_fam):
            parts.append(f"[{fam}] " + "; ".join(by_fam[fam]))
        dip_new = "; ".join(parts)
        seeds_seen = sorted(set(
            int(x) for x in dip["n_seeds"].dropna().unique()))
        seed_note = f" (seed counts present: {seeds_seen})"
    else:
        dip_new, seed_note = PENDING, ""

    rows.append({
        "item": "Phase 4 / reviewer AbH52 #3 + roadmap T2.3",
        "question": "Is the per-round acceptance distribution bimodal, "
                    "tested rather than asserted via a zero/nonzero split?",
        "old": "claim rested on a zero/nonzero split; no test statistic",
        "new": f"Hartigan dip test: {dip_new}{seed_note} "
               "-> results/mlsys/dip_test.csv, dip_by_context.csv",
        "read": "Small p-values reject unimodality. Reported alongside "
                "p_zero so the split is descriptive, not load-bearing.",
    })
    rows.append({
        "item": "Phase 4 / reviewer AbH52 #1",
        "question": "Do the tables report alpha as accepted-prefix/gamma "
                    "(per-round), not the i.i.d. per-token parameter?",
        "old": "unverified",
        "new": _acceptance_verdict(REPO),
        "read": "acceptance_rate is SUM(n_acc)/(R*gamma). The i.i.d. "
                "parameter (alpha_iid) is a different number and is "
                "reported separately with its KS distance.",
    })

    # ---- Phase 5: vLLM --------------------------------------------------
    rows.append({
        "item": "Phase 5 / roadmap T2.1 (no production-stack baseline)",
        "question": "How does RASD compare to vLLM at 128k on the same "
                    "hardware, in the same units?",
        "old": "no vLLM baseline; speedups had no external reference point",
        "new": _vllm_summary(vllm),
        "read": "Compare vLLM throughput_tps_end_to_end against RASD "
                "throughput_tps (both end-to-end generation tok/s). Rows "
                "with unit_matched=no must be excluded from the ratio.",
    })

    return rows


def _count(d: Path, suffix: str) -> int:
    return len(list(d.glob(f"*.{suffix}"))) if d.is_dir() else 0


def _bool_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    """(n_true, n_false, n_unchecked) for an `ok` column after a CSV
    round-trip.

    Round-tripping a column holding True/False/None through CSV yields
    *strings*, so `df["ok"] == True` silently reports zero. Normalise
    explicitly instead of relying on dtype.
    """
    if "ok" not in df.columns:
        return 0, 0, 0
    norm = df["ok"].astype(str).str.strip().str.lower()
    n_true = int((norm == "true").sum())
    n_false = int((norm == "false").sum())
    n_unchecked = int(len(df) - n_true - n_false)
    return n_true, n_false, n_unchecked


def _acceptance_verdict(repo: Path) -> str:
    for name in ("acceptance_accounting.csv",
                 "acceptance_accounting_seed42_pg19.csv"):
        df = _read(repo / "results/mlsys" / name)
        if df is None or df.empty:
            continue
        n_true, n_false, n_none = _bool_counts(df)
        if n_true + n_false == 0:
            continue
        src = "acceptance_accounting.csv" if name.startswith("acceptance_accounting.csv") else name
        return (f"{n_true} rows verified as per-round alpha, "
                f"{n_false} mismatched, {n_none} unverifiable "
                f"-> results/mlsys/{src}")
    return PENDING


def _vllm_summary(vllm: pd.DataFrame | None) -> str:
    if vllm is None or vllm.empty:
        return PENDING
    ok = vllm[vllm["status"] == "ok"]
    if ok.empty:
        return (f"all {len(vllm)} cell(s) failed: "
                + ", ".join(f"{r['model'].split('/')[-1]}={r['status']}"
                            for _, r in vllm.iterrows()))
    parts = []
    for _, r in ok.iterrows():
        # The rung and the model length it ACTUALLY ran at: the ladder's 64k
        # fallback is a legitimate row, but it is not the 128k rung and the
        # reader has to be able to see that from the number itself.
        ran = r.get("max_model_len")
        ran_txt = (f", ran at {int(ran)}" if ran == ran and ran else "")
        parts.append(f"{r['model'].split('/')[-1]}@"
                     f"{int(r['context_length'])}: "
                     f"{r['throughput_tps_end_to_end']} tok/s "
                     f"(end-to-end{ran_txt}, "
                     f"unit_matched={r['unit_matched']})")
    return "; ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "results/mlsys/MASTER_TABLE.txt"))
    args = ap.parse_args()

    rows = build_rows()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 78)
    lines.append("MLSys experiment program — MASTER TABLE (old vs new)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Generated by scripts/mlsys_master_table.py.")
    lines.append("'old' numbers are read from committed CSVs (file named inline).")
    lines.append("'new' numbers come only from results/mlsys/; cells the pod has")
    lines.append("not produced yet read '(pending GPU run)' — they are never")
    lines.append("back-filled with the old value or with an estimate.")
    lines.append("")
    for i, r in enumerate(rows, 1):
        lines.append("-" * 78)
        lines.append(f"[{i}] {r['item']}")
        lines.append(f"    question : {r['question']}")
        lines.append(f"    OLD      : {r['old']}")
        lines.append(f"    NEW      : {r['new']}")
        lines.append(f"    read     : {r['read']}")
    lines.append("-" * 78)
    lines.append("")
    lines.append("NOTE ON SEED COUNTS: any 'old' cell marked single-seed is")
    lines.append("exactly the weakness these runs exist to remove. Do not")
    lines.append("present a single-seed 'new' cell as multi-seed evidence.")
    lines.append("")

    text = "\n".join(lines)
    out.write_text(text)
    print(text)
    print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

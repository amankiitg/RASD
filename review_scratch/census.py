#!/usr/bin/env python3
"""F2 degeneracy census, re-derived row by row (reviewer code).

For each row the repo's Fig C counts, find its session CSV row (temperature, status) and
its token sidecar, rescore with src/analysis/repetition.py, and with an independent
periodicity check written here. Run from the repo root with any python3 (no deps).
"""
import collections, csv, glob, json, os, re, sys
sys.path.insert(0, os.getcwd())
from src.analysis.repetition import row_stats, row_reasons

R = "results/mlsys/"


def load(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# index every session row by run_id -> (temperature, csv)
ROW = {}
for f in glob.glob(R + "**/session_a_*.csv", recursive=True):
    try:
        for r in load(f):
            if r.get("run_id"):
                ROW.setdefault(r["run_id"], []).append((r.get("temperature"), r.get("status"), f))
    except Exception:
        pass

SIDE = {}
for p in glob.glob(R + "**/*.json", recursive=True):
    if "/tokens" not in p and "arms_tokens" not in p:
        continue
    try:
        b = json.load(open(p))
    except Exception:
        continue
    ids = b.get("generated_token_ids") or b.get("generated_ids")
    if ids and b.get("run_id"):
        SIDE.setdefault(b["run_id"], []).append((p, ids))


def my_period(ids, maxp=64, tail_min=0.5):
    """Independent check: does the generated tail settle into an exact cycle?
    For each period p, walk back from the end while ids[i]==ids[i+p]; report the p
    that covers the longest tail. A row 'loops' if that tail covers >= tail_min of the
    row with at least 3 full repetitions; also report the longest exact repeat."""
    n = len(ids)
    best = (0, 0)
    for p in range(1, min(maxp, n // 3) + 1):
        i = n - 1 - p
        while i >= 0 and ids[i] == ids[i + p]:
            i -= 1
        matched = (n - p) - (i + 1)          # positions j with ids[j]==ids[j+p] in the tail run
        tail = matched + p if matched >= p else 0   # at least one full repeat, else no cycle
        if tail > best[1]:
            best = (p, tail)
    p, tail = best
    loops = p > 0 and tail / n >= tail_min and tail >= 3 * p  # >=3 full periods
    # share of distinct 8-grams (low = repetitive), independent of the 16-gram rule
    g = [tuple(ids[i:i + 8]) for i in range(n - 7)]
    distinct = len(set(g)) / len(g) if g else 1.0
    return dict(my_period=p, my_tail=round(tail / n, 3), my_loop=loops, distinct8=round(distinct, 3))


def score(rid):
    s = SIDE.get(rid)
    if not s:
        return None
    ids = s[0][1]
    st = row_stats(ids)
    return dict(n=len(ids), repo_loop=bool(row_reasons(st)), reasons=row_reasons(st),
                tail=st["rep_periodic_tail_share"], in16=st.get("rep_tokens_in_repeat"),
                span=st.get("rep_longest_repeat_span"), **my_period(ids))


def show(title, rids, flags=None):
    print("=" * 78, f"\n{title}: {len(rids)} rows")
    loops_repo = loops_mine = 0
    temps = collections.Counter()
    for rid in rids:
        meta = ROW.get(rid, [("?", "?", "?")])
        temps[meta[0][0]] += 1
        sc = score(rid)
        flag = flags.get(rid) if flags else None
        if sc:
            loops_repo += sc["repo_loop"]; loops_mine += sc["my_loop"]
        mark = ""
        if sc and (sc["repo_loop"] or sc["my_loop"] or flag):
            mark = "  <== " + ("; ".join(sc["reasons"]) if sc["reasons"] else "")
        print(f"  {rid[:58]:58s} T={meta[0][0]:>4} csv_flag={flag} "
              + (f"repo={sc['repo_loop']!s:5} tail={sc['tail']:.2f} in16={sc['in16']:.2f} span={sc['span']:>3} "
                 f"| mine p={sc['my_period']:>2} tail={sc['my_tail']:.2f} d8={sc['distinct8']:.2f} loop={sc['my_loop']}"
                 if sc else "NO SIDECAR") + mark)
    print(f"  temperatures: {dict(temps)}  repo-rule loops (sidecar): {loops_repo}  my-rule loops: {loops_mine}")


# --- group 1: first-window audit (CSV verdicts only)
fw = load(R + "natural_f1_128k_first_window.csv")
show("GREEDY first-window audit (natural_f1_128k_first_window.csv)", [r["run_id"] for r in fw],
     {r["run_id"]: r["row_flag"] for r in fw})

# --- group 2: temperature arms
tarms = []
for f in ("seeds_ablations/repetition_SA_TEMP_0_128K_SPEC.csv", "seeds_ablations/repetition_SA_TEMP_1_0_128K_SPEC.csv"):
    tarms += load(R + f)
show("'GREEDY' temperature arms (repetition_SA_TEMP_{0,1_0})", [r["run_id"] for r in tarms],
     {r["run_id"]: r.get("rep_degenerate") for r in tarms})

# --- group 3: 'sampled' = what lcfm_figures.py globs
samp = []
for g in ("sampling_session/repetition_*.csv", "n5_extension/repetition_*.csv", "fp4_seeds/repetition_*.csv"):
    for f in sorted(glob.glob(R + g)):
        for r in load(f):
            if r.get("status") and r["status"] != "ok":
                continue
            if r.get("rep_periodic_tail_share") in (None, ""):
                continue
            samp.append((r["run_id"], r.get("rep_degenerate"), f))
for g in ("rerun_r1/session_a_*.csv", "rerun_r2/session_a_*.csv", "decode_df/session_a_*.csv"):
    for f in sorted(glob.glob(R + g)):
        for r in load(f):
            if r.get("status") == "ok" and SIDE.get(r["run_id"]):
                samp.append((r["run_id"], "scored", f))
show("'SAMPLED T=0.7' census as lcfm_figures.py builds it", [x[0] for x in samp],
     {x[0]: x[1] for x in samp})
c = collections.Counter(x[0] for x in samp)
print("  duplicate run_ids in census:", {k: v for k, v in c.items() if v > 1})

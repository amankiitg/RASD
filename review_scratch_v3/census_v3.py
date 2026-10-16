#!/usr/bin/env python3
"""F2 census for v3, grouped by each row's OWN recorded temperature (reviewer code).

The temperature of a row is read from the session CSV in the SAME directory as the
repetition CSV / sidecar that carries its verdict (run_ids repeat across sessions, e.g.
RR_SANITY_*), never from the stage name. Every row is rescored from its sidecar where one
is released, with src/analysis/repetition.py and with an independent periodicity check.
Run from the repo root: python3 review_scratch_v3/census_v3.py
"""
import collections, csv, glob, json, os, sys
sys.path.insert(0, os.getcwd())
from src.analysis.repetition import row_stats, row_reasons

R = "results/mlsys/"


def load(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def session_rows(d):
    out = {}
    for f in glob.glob(os.path.join(d, "session_a_*.csv")):
        for r in load(f):
            if r.get("run_id"):
                out.setdefault(r["run_id"], r)
    return out


def sidecar(d, rid):
    for sub in ("tokens", "tokens_slim"):
        for p in glob.glob(os.path.join(d, sub, rid + ".json")):
            b = json.load(open(p))
            ids = b.get("generated_token_ids")
            if ids:
                return ids
    return None


def periodic(ids, maxp=64):
    """Independent: longest exact-cycle tail with >= 3 full periods; loop if >= 50% of row."""
    n, best = len(ids), (0, 0)
    for p in range(1, min(maxp, n // 3) + 1):
        i = n - 1 - p
        while i >= 0 and ids[i] == ids[i + p]:
            i -= 1
        matched = (n - p) - (i + 1)
        tail = matched + p if matched >= p else 0
        if tail > best[1]:
            best = (p, tail)
    p, tail = best
    return p > 0 and tail >= 3 * p and tail / n >= 0.5


rows = []  # (run_id, temperature, tokens, verdict_from_csv, repo_rescore, my_periodic, source)
# first-window audit: verdict CSV; temperature from its own session CSV (pod_results)
fw_sess = {r["run_id"]: r for r in load(R + "incident_20261010T113537Z_repetition_stop/pod_results/natural_f1_128k.csv")}
for r in load(R + "natural_f1_128k_first_window.csv"):
    s = fw_sess[r["run_id"]]
    rows.append((r["run_id"], float(s["temperature"]), int(s["tokens_generated"]),
                 r["row_flag"].strip().upper() == "LOOP", None, None, "first-window"))
# every repetition CSV the paper's census can draw on, plus fixed-engine sidecars
for d in ("seeds_ablations", "sampling_session", "n5_extension", "fp4_seeds"):
    sess = session_rows(R + d)
    for f in glob.glob(R + d + "/repetition_*.csv"):
        stage = os.path.basename(f)
        if d == "seeds_ablations" and "SA_TEMP" not in stage:
            continue  # the paper's census uses only the temperature arm from this session
        for r in load(f):
            if r.get("status") and r["status"] != "ok":
                continue
            if r.get("rep_periodic_tail_share") in (None, ""):
                continue
            s = sess.get(r["run_id"])
            ids = sidecar(R + d, r["run_id"])
            rep = bool(row_reasons(row_stats(ids))) if ids else None
            rows.append((r["run_id"], float(s["temperature"]), int(s["tokens_generated"]),
                         str(r["rep_degenerate"]).lower() == "true", rep,
                         periodic(ids) if ids else None, d))
for d in ("rerun_r1", "rerun_r2"):
    for rid, s in session_rows(R + d).items():
        if s.get("status") != "ok":
            continue
        ids = sidecar(R + d, rid)
        if not ids:
            continue
        rep = bool(row_reasons(row_stats(ids)))
        rows.append((rid, float(s["temperature"]), int(s["tokens_generated"]), rep, rep, periodic(ids), d))

groups = collections.defaultdict(list)
for x in rows:
    key = ("first-window" if x[6] == "first-window" else
           "SA_TEMP" if x[0].startswith("SA_TEMP") else "pooled", x[1])
    groups[key].append(x)
for key in sorted(groups):
    g = groups[key]
    loops = [x for x in g if x[3]]
    plumb = sum(1 for x in g if "GATE" in x[0] or "SANITY" in x[0] or "SAMP_CHECK" in x[0])
    uniq = len(set(x[0] for x in g))
    resc = [x for x in g if x[4] is not None]
    print(f"{key[0]:13s} T={key[1]:<4} rows={len(g):3d} unique_ids={uniq:3d} plumbing/repeat={plumb:2d} "
          f"tokens={sorted(set(x[2] for x in g))} loops(csv verdict)={len(loops)} -> {[x[0] for x in loops]}")
    print(f"{'':13s}        rescored from sidecar: {len(resc)} rows, repo-rule loops {sum(1 for x in resc if x[4])}, "
          f"my periodic loops {sum(1 for x in resc if x[5])}; rows without sidecar: {len(g)-len(resc)}")
    disagree = [x[0] for x in resc if x[4] != x[3]]
    if disagree:
        print(f"{'':13s}        CSV verdict vs sidecar rescore DISAGREE on: {disagree}")

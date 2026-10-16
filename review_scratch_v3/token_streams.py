"""Compare bw_diag token streams across wait modes and sessions (reviewer code)."""
import json, csv, glob, itertools, os
j = json.load(open("results/mlsys/bw_diag_token_streams.json"))
groups = {}
for k, v in j.items():
    key = (v["context_length"], v["arm"], v["mode"])
    groups.setdefault(key, []).append((k, v["tokens"]))
temp = {}
for f in glob.glob("results/mlsys/bw_diag_p*/*/*/df_arms.csv"):
    for r in csv.DictReader(open(f)):
        temp[(f.split("/")[2], f.split("/")[3], f.split("/")[4], r["arm"], r["mode"], r["context_length"])] = (r["temperature"], r["status"])
for key, items in sorted(groups.items()):
    print(key, f"{len(items)} streams")
    for (a, ta), (b, tb) in itertools.combinations(items, 2):
        n = min(len(ta), len(tb))
        div = next((i for i in range(n) if ta[i] != tb[i]), None)
        print(f"   {a.split('/tokens')[0]:22s} vs {b.split('/tokens')[0]:22s} len {len(ta)}/{len(tb)} "
              + ("IDENTICAL" if div is None and len(ta) == len(tb) else f"diverge at index {div}"))
print("recorded temperatures of bw_diag rows:")
for k, v in sorted(temp.items()):
    print("  ", k, v)

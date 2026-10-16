"""F5 on matched books and the paired 1M acceptance change (reviewer code)."""
import csv, glob, re, statistics as st, math
R = "results/mlsys/"
def rows(gs):
    out = []
    for g in gs:
        for f in glob.glob(R + g):
            for r in csv.DictReader(open(f, encoding="utf-8-sig")):
                if r.get("status") == "ok" and "GATE" not in r["run_id"] and "SANITY" not in r["run_id"]:
                    out.append(r)
    return out
dec = lambda r: (float(r["time_sec"]) - float(r["ttft_ms"]) / 1000) / (int(r["tokens_generated"]) - 1)
bk = lambda r: re.search(r"pg19_train_\d+", r["run_id"]).group(0)
new = rows(["rerun_r1/session_a_*.csv", "rerun_r2/session_a_*.csv", "rerun_topup/session_a_*.csv"])
for ctx in (131072, 524288, 1048576):
    sp = {bk(r): dec(r) for r in new if int(r["context_length"]) == ctx and r["spec_steps"] == "4" and r["seed"] == "42"}
    tg = {bk(r): dec(r) for r in new if int(r["context_length"]) == ctx and r["spec_steps"] == "0"}
    common = sorted(set(sp) & set(tg))
    per = {b: round(tg[b] / sp[b], 3) for b in common}
    print(ctx, "matched books", common, "per-book target/spec", per,
          "ratio of means", round(st.mean(tg[b] for b in common) / st.mean(sp[b] for b in common), 4))
old = rows(["sampling_session/session_a_SAMP_1M_SPEC.csv", "fp4_seeds/session_a_FP4_SEED_1M_SPEC.csv"])
oa = {(r["seed"], bk(r)): float(r["acceptance_rate"]) for r in old}
na = {(r["seed"], bk(r)): float(r["acceptance_rate"]) for r in new if int(r["context_length"]) == 1048576 and r["spec_steps"] == "4"}
k = sorted(set(oa) & set(na))
d = [na[x] - oa[x] for x in k]
rel = [na[x] / oa[x] - 1 for x in k]
m, s = st.mean(d), st.stdev(d)
print(f"1M acceptance paired over {len(k)} cells: mean diff {m:+.4f} sd {s:.4f} se {s/math.sqrt(len(k)):.4f} "
      f"~95% t-interval [{m-2.179*s/math.sqrt(len(k)):+.4f}, {m+2.179*s/math.sqrt(len(k)):+.4f}] "
      f"(old mean {st.mean(oa[x] for x in k):.4f}); per-cell relative change range "
      f"{100*min(rel):+.1f}% .. {100*max(rel):+.1f}%")

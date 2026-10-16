"""v5-only checks (reviewer code): 128k TTFT per configuration, memory rows, bench rank spread."""
import csv, glob, json
R = "results/mlsys/"
load = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
print("== 128k first-token time by configuration (bw_diag arms + split-KV campaign target rows)")
cfg = {}
for f in glob.glob(R + "bw_diag_p*/*/*/df_arms.csv"):
    for r in load(f):
        if int(r["context_length"]) == 131072:
            k = (r["arm"], r["kv_block_size"], f.split("/")[4], r["mode"])
            cfg.setdefault(k, []).append((round(float(r["ttft_ms"]) / 1000, 2), f.split("/")[2]))
for f in glob.glob(R + "rerun_r*/session_a_*.csv") + glob.glob(R + "rerun_topup/session_a_*.csv"):
    for r in load(f):
        if r["status"] == "ok" and int(r["context_length"]) == 131072:
            k = ("splitkv-campaign", r["kv_block_size"], "blocking", "spec" if r["spec_steps"] == "4" else "target")
            cfg.setdefault(k, []).append((round(float(r["ttft_ms"]) / 1000, 2), f.split("/")[2]))
for k, v in sorted(cfg.items()):
    print("  ", k, sorted(v))
print("== memory")
camp = [r for f in glob.glob(R + "rerun_r1/session_a_RR_1M_*.csv") + glob.glob(R + "rerun_r2/session_a_*.csv")
        + glob.glob(R + "rerun_topup/session_a_TP_1M*.csv") for r in load(f)
        if r["status"] == "ok" and int(r["context_length"]) == 1048576]
print(f"  split-KV 1M rows n={len(camp)} rank-0 max {max(float(r['gpu_peak_mem_mb']) for r in camp)/1024:.2f} GiB")
print("== bench rank spread: JSON skew entries are (wait, sync) per label")
for m in ("blocking", "async"):
    sk = json.load(open(R + f"bw_diag_p1/bench/ringbench_{m}.json"))["skew"]
    wait = [sk[2 * i]["spread_ms"] for i in range(len(sk) // 2)]
    sync = [sk[2 * i + 1]["spread_ms"] for i in range(len(sk) // 2)]
    print(f"  {m}: wait-timing spread max {max(wait):.4f}  post-sync spread max {max(sync):.4f}")

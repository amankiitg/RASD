#!/usr/bin/env python3
"""Independent recompute of the v3 blocking-wait numbers and other new v3 figures (reviewer code).
Run from the repo root: python3 review_scratch_v3/bw_recompute.py
"""
import csv, glob, statistics as st

R = "results/mlsys/"


def load(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


dec = lambda r: (float(r["time_sec"]) - float(r["ttft_ms"]) / 1000) / (int(r["tokens_generated"]) - 1)

print("== bench (bw_diag_p1)")
b = {(r["label"], r["timing"]): r for r in load(R + "bw_diag_p1/bench/ringbench_blocking.csv")}
a = {(r["label"], r["timing"]): r for r in load(R + "bw_diag_p1/bench/ringbench_async.csv")}
labels = ["tiny_4kb", "mb33", "k8_66mb_expand_after", "mb133", "mb266"]
for L in labels:
    bw, bs, as_ = float(b[(L, "wait")]["ms_per_round"]), float(b[(L, "sync")]["ms_per_round"]), float(a[(L, "sync")]["ms_per_round"])
    mb = float(b[(L, "wait")]["size_mb_actual"])
    print(f"  {L:22s} {mb:8.3f} MB  blocking-wait {bw:.4f}  blocking-postsync {bs:.4f}  async-postsync {as_:.4f}  "
          f"postsync diff {100*(bs/as_-1):+.1f}%  async GB/s {mb/1000/(as_/1000):.2f}")
bws = [float(b[(L, "wait")]["ms_per_round"]) for L in labels]
print(f"  blocking range {min(bws):.4f}-{max(bws):.4f}, spread {100*(max(bws)/min(bws)-1):.2f}%; "
      f"payload range {float(b[('mb266','wait')]['size_mb_actual'])/float(b[('tiny_4kb','wait')]['size_mb_actual']):.0f}x")
pred_4k = 224 * (float(b[("tiny_4kb", "wait")]["ms_per_round"]) - float(a[("mb266", "sync")]["ms_per_round"])) / 1000
pred_266 = 224 * (float(b[("mb266", "wait")]["ms_per_round"]) - float(a[("mb266", "sync")]["ms_per_round"])) / 1000
print(f"  prediction 224*(blocking 4KB - async 266MB) = {pred_4k:.4f} s; same-payload 224*(10.3520-2.6191) = {pred_266:.4f} s")

print("== engine arms")
arm = {}
for f in glob.glob(R + "bw_diag_p*/*/*/df_arms.csv"):
    sess, stage, mode = f.split("/")[2:5]
    for r in load(f):
        arm[(sess, stage, mode, r["arm"], r["mode"], int(r["context_length"]))] = r
def g(*k):
    return dec(arm[k])
for sess in ("bw_diag_p1", "bw_diag_p2"):
    bl, asy = g(sess, "arms", "blocking", "ring2048", "target", 131072), g(sess, "arms", "async", "ring2048", "target", 131072)
    print(f"  {sess} 128k ring2048 target: blocking {bl:.4f} async {asy:.4f} saving {bl-asy:.4f} s; "
          f"vs prediction {pred_4k:.4f}: {100*(bl-asy-pred_4k)/(bl-asy):+.1f}% ; "
          f"split-KV (same session, blocking) {g(sess,'arms','blocking','splitkv','target',131072):.4f} -> "
          f"ring-minus-splitKV {bl-g(sess,'arms','blocking','splitkv','target',131072):.4f} s; poll share {100*(bl-asy)/(bl-g(sess,'arms','blocking','splitkv','target',131072)):.0f}%")
headline_removed_128k = 3.4925 - 1.0596
print(f"  headline removed at 128k {headline_removed_128k:.4f} s; p1 poll saving / that = {100*(g('bw_diag_p1','arms','blocking','ring2048','target',131072)-g('bw_diag_p1','arms','async','ring2048','target',131072))/headline_removed_128k:.0f}%")
b1, a1 = g("bw_diag_p2", "onem", "blocking", "ring2048", "target", 1048576), g("bw_diag_p2", "onem", "async", "ring2048", "target", 1048576)
print(f"  1M ring2048 target: blocking {b1:.4f} async {a1:.4f} change {100*(a1/b1-1):+.2f}% ; saving {b1-a1:.3f} s "
      f"= {100*(b1-a1)/(17.0902-3.2107):.1f}% of the headline 1M removed time")
xfer_1m = 224 * (2 * 32 * (1048576 // 8) * 128 * 2) / 101.69e9
print(f"  1M transfer at the bench's 101.7 GB/s: 224 rounds x 2.15 GB = {xfer_1m:.2f} s of the 13.88 s removed")
print("== memory (worst rank)")
for k, r in sorted(arm.items()):
    if k[5] == 1048576:
        print(f"  {k}: alloc {float(r['peak_alloc_mb_worst_rank'])/1024:.2f} GiB reserved {float(r['peak_reserved_mb_worst_rank'])/1024:.2f} GiB "
              f"mode={r['ring_decode_mode']} commit={r['git_commit'][:8]}")
camp = [r for f in glob.glob(R + "rerun_r1/session_a_RR_1M_*.csv") + glob.glob(R + "rerun_r2/session_a_*.csv") for r in load(f)
        if r["status"] == "ok" and int(r["context_length"]) == 1048576]
print(f"  split-KV campaign 1M rows: n={len(camp)} rank-0 gpu_peak_mem_mb max {max(float(r['gpu_peak_mem_mb']) for r in camp)/1024:.2f} GiB "
      f"(worst-rank columns present: {'peak_reserved_mb_worst_rank' in camp[0]})")

print("== 128k first-token times (s), every ok 128k row with prompt 130047/130048")
tt = []
for f in glob.glob(R + "**/*.csv", recursive=True):
    try:
        rs = load(f)
    except Exception:
        continue
    for r in rs:
        try:
            if r.get("status") == "ok" and int(r.get("context_length") or 0) == 131072 and r.get("ttft_ms") \
                    and "Llama-3-8B-Instruct-Gradient" in r.get("target_model_name", "Gradient"):
                tt.append((float(r["ttft_ms"]) / 1000, f.replace(R, ""), r["run_id"]))
        except Exception:
            pass
tt.sort()
print(f"  n={len(tt)} min {tt[0][0]:.2f} ({tt[0][1]}) max {tt[-1][0]:.2f} ({tt[-1][1]})")
print("  RR_128K_TARGET:", [round(x[0], 2) for x in tt if x[2].startswith("RR_128K_TARGET")])

print("== F5 end-to-end, matched book at 128k (train_0)")
def rows(gs):
    return [r for gg in gs for f in glob.glob(R + gg) for r in load(f) if r["status"] == "ok" and "GATE" not in r["run_id"] and "SANITY" not in r["run_id"]]
new = rows(["rerun_r1/session_a_*.csv", "rerun_r2/session_a_*.csv", "rerun_topup/session_a_*.csv"])
for ctx, books in ((131072, ["pg19_train_0"]),):
    sp = [r for r in new if int(r["context_length"]) == ctx and r["spec_steps"] == "4" and r["seed"] == "42" and any(b + "_" in r["run_id"] for b in books)]
    tg = [r for r in new if int(r["context_length"]) == ctx and r["spec_steps"] == "0" and any(b + "_" in r["run_id"] for b in books)]
    tps = lambda rs: st.mean(int(r["tokens_generated"]) / float(r["time_sec"]) for r in rs)
    print(f"  128k: spec {len(sp)} target {len(tg)}  e2e tps ratio {tps(sp)/tps(tg):.4f}")
print("== 64-token rows in the T=0.7 census")
for d in ("sampling_session", "n5_extension", "fp4_seeds"):
    for f in glob.glob(R + d + "/session_a_*.csv"):
        for r in load(f):
            if r.get("tokens_generated") == "64":
                print("  ", d, r["run_id"], r["temperature"])

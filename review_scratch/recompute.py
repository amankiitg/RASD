#!/usr/bin/env python3
"""Independent recompute of every number in manuscript/lcfm/main.tex (reviewer code).

Written from the brief's definitions, NOT from scripts/lcfm_figures.py:
  decode-only s/token = (time_sec - ttft_ms/1000) / (tokens_generated - 1)
  matched cell = (seed, book); run_ids containing GATE or SANITY excluded.
Run from the repo root:  python3 review_scratch/recompute.py
"""
import csv, glob, json, os, re, statistics as st, sys

R = "results/mlsys/"
OUT = {}


def load(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def ok(rows):
    return [r for r in rows if r.get("status") == "ok"]


def plumbing(rid):
    return "GATE" in rid or "SANITY" in rid


def book(rid):
    return re.search(r"pg19_train_\d+", rid).group(0)


def dec(r):
    return (float(r["time_sec"]) - float(r["ttft_ms"]) / 1000) / (int(r["tokens_generated"]) - 1)


def e2e_spt(r):
    return float(r["time_sec"]) / int(r["tokens_generated"])


def cells(files, ctx, spec, val=dec):
    c = {}
    for f in files:
        for r in ok(load(f)):
            if plumbing(r["run_id"]) or int(r["context_length"]) != ctx or int(r["spec_steps"]) != spec:
                continue
            c.setdefault((int(r["seed"]), book(r["run_id"])), []).append(val(r))
    return {k: st.mean(v) for k, v in c.items()}, c


# ------------------------------------------------------------------ pools (chosen by me)
OLD = {
    (131072, 4): [R + "sampling_session/session_a_SAMP_ALIGNED_128K_SPEC.csv",
                  R + "n5_extension/session_a_SAMP_EXT_ALIGNED_128K_SPEC.csv"],
    (131072, 0): [R + "sampling_session/session_a_SAMP_ALIGNED_128K_TARGET.csv"],
    (524288, 4): [R + "sampling_session/session_a_SAMP_ALIGNED_512K_SPEC.csv",
                  R + "n5_extension/session_a_SAMP_EXT_ALIGNED_512K_SPEC.csv"],
    (524288, 0): [R + "sampling_session/session_a_SAMP_ALIGNED_512K_TARGET.csv"],
    (1048576, 4): [R + "sampling_session/session_a_SAMP_1M_SPEC.csv",
                   R + "fp4_seeds/session_a_FP4_SEED_1M_SPEC.csv"],
    (1048576, 0): [R + "sampling_session/session_a_SAMP_1M_TARGET.csv"],
}
NEW = sorted(glob.glob(R + "rerun_r1/session_a_*.csv") + glob.glob(R + "rerun_r2/session_a_*.csv"))
TOPUP = sorted(glob.glob(R + "rerun_topup/session_a_*.csv"))
CT = {131072: "128k", 524288: "512k", 1048576: "1M"}

print("=" * 70, "\n1. HEADLINE matched cells (decode-only)")
head = {}
for ctx in CT:
    for spec, arm in ((4, "spec"), (0, "target")):
        o, oraw = cells(OLD[(ctx, spec)], ctx, spec)
        n, nraw = cells(NEW, ctx, spec)
        keys = sorted(set(o) & set(n))
        dup = {k: len(v) for k, v in {**oraw}.items() if len(v) > 1}
        om, nm = st.mean(o[k] for k in keys), st.mean(n[k] for k in keys)
        head[(ctx, arm)] = dict(keys=keys, old=om, new=nm, sp=om / nm,
                                old_only=sorted(set(o) - set(n)), new_only=sorted(set(n) - set(o)))
        print(f"  {CT[ctx]:5s} {arm:6s} old {om:.4f} new {nm:.4f} speedup {om/nm:.4f} n={len(keys)} "
              f"dupcells_old={dup} old_only={sorted(set(o)-set(n))} new_only={sorted(set(n)-set(o))}")
        print("        cells:", keys)
OUT["head"] = {f"{CT[c]}_{a}": {k: v for k, v in d.items() if k in ("old", "new", "sp")} | {"n": len(d["keys"])}
               for (c, a), d in head.items()}

# Same-cell mixture check: spec 1M with only seed 42
o, _ = cells(OLD[(1048576, 4)], 1048576, 4); n, _ = cells(NEW, 1048576, 4)
for s in (42, 123, 456):
    ks = [k for k in set(o) & set(n) if k[0] == s]
    print(f"  1M spec seed {s}: n={len(ks)} old {st.mean(o[k] for k in ks):.4f} new {st.mean(n[k] for k in ks):.4f}"
          f" speedup {st.mean(o[k] for k in ks)/st.mean(n[k] for k in ks):.3f}")

print("=" * 70, "\n2. F5 spec over target inside fixed engine (decode-only and e2e)")
for ctx in CT:
    sp, _ = cells(NEW, ctx, 4)
    tg, _ = cells(NEW + TOPUP, ctx, 0)
    tg_nt, _ = cells(NEW, ctx, 0)
    sp_e, _ = cells(NEW, ctx, 4, e2e_spt)
    tg_e, _ = cells(NEW + TOPUP, ctx, 0, e2e_spt)
    # seed-42-only spec (books), since target-only is seed 42 only
    sp42 = {k: v for k, v in sp.items() if k[0] == 42}
    sp42e = {k: v for k, v in sp_e.items() if k[0] == 42}
    print(f"  {CT[ctx]}: spec cells {len(sp)} (seed42 books {len(sp42)}) target cells(with topup) {len(tg)} "
          f"[without topup {len(tg_nt)}]")
    print(f"     decode: spec(all) {st.mean(sp.values()):.4f}  spec(s42) {st.mean(sp42.values()):.4f}  "
          f"target {st.mean(tg.values()):.4f} (no-topup {st.mean(tg_nt.values()):.4f})")
    print(f"     ratio target/spec: all-spec {st.mean(tg.values())/st.mean(sp.values()):.4f}  "
          f"s42-spec {st.mean(tg.values())/st.mean(sp42.values()):.4f}  "
          f"no-topup/all {st.mean(tg_nt.values())/st.mean(sp.values()):.4f}")
    print(f"     e2e s/tok ratio target/spec: all {st.mean(tg_e.values())/st.mean(sp_e.values()):.4f}  "
          f"s42 {st.mean(tg_e.values())/st.mean(sp42e.values()):.4f}")
    # throughput-based e2e (tokens / time) as the appendix says throughput = tokens over time
    tps = lambda files, spec, seed=None: st.mean(
        int(r["tokens_generated"]) / float(r["time_sec"]) for f in files for r in ok(load(f))
        if not plumbing(r["run_id"]) and int(r["context_length"]) == ctx and int(r["spec_steps"]) == spec
        and (seed is None or int(r["seed"]) == seed))
    print(f"     e2e mean-tps ratio spec/target: all {tps(NEW,4)/tps(NEW+TOPUP,0):.4f}  s42 {tps(NEW,4,42)/tps(NEW+TOPUP,0):.4f}")
    bk = sorted(tg.values())
    print(f"     target per-book: {[(k[1], round(v, 4)) for k, v in sorted(tg.items())]} spread max/min {bk[-1]/bk[0]:.4f}")

# 1M row wall split
rows1m = [r for f in NEW for r in ok(load(f)) if int(r["context_length"]) == 1048576 and not plumbing(r["run_id"])]
for spec in (4, 0):
    rr = [r for r in rows1m if int(r["spec_steps"]) == spec]
    print(f"  1M spec={spec} mean time_sec {st.mean(float(r['time_sec']) for r in rr):.1f}  "
          f"mean ttft {st.mean(float(r['ttft_ms']) for r in rr)/1000:.1f} s (n={len(rr)})")
rr = [r for f in NEW + TOPUP for r in ok(load(f)) if int(r["context_length"]) == 1048576 and not plumbing(r["run_id"])]
print(f"  1M all rows incl topup: time {st.mean(float(r['time_sec']) for r in rr):.1f} ttft {st.mean(float(r['ttft_ms']) for r in rr)/1000:.1f} n={len(rr)}")

# seed spread inside fixed engine (1M spec, per seed mean over 5 books)
sp, _ = cells(NEW, 1048576, 4)
ps = {s: st.mean(v for k, v in sp.items() if k[0] == s) for s in (42, 123, 456)}
print(f"  fixed-engine 1M spec per-seed decode means {ps}  spread {max(ps.values())/min(ps.values()):.4f}")
pb = {}
for k, v in sp.items():
    pb.setdefault(k[1], []).append(v)
pbm = {b: st.mean(v) for b, v in pb.items()}
print(f"  fixed-engine 1M spec per-book means {pbm} spread {max(pbm.values())/min(pbm.values()):.4f}")

print("=" * 70, "\n3. Topup table and means")
for f in TOPUP:
    for r in ok(load(f)):
        print(f"  {r['run_id']:40s} {dec(r):.4f} {r['ring_decode_mode']} L={r['ring_layers_patched']} commit={r['git_commit'][:8]}")
for ctx in (524288, 1048576):
    a, _ = cells(NEW, ctx, 0); b, _ = cells(NEW + TOPUP, ctx, 0)
    print(f"  {CT[ctx]} target mean: before {st.mean(a.values()):.4f} (n={len(a)}) after {st.mean(b.values()):.4f} (n={len(b)}) "
          f"thin/wide {st.mean(a.values())/st.mean(b.values()):.4f}")
# r1 TARGET 128k and GATE rows, for the topup table's 128k line
for f in [R + "rerun_r1/session_a_RR_128K_TARGET.csv", R + "rerun_r1/session_a_RR_GATE_128K_TARGET.csv"]:
    for r in ok(load(f)):
        print(f"  {r['run_id']:40s} {dec(r):.4f}")

print("=" * 70, "\n4. Seed-corrected acceptance at 1M (13 matched cells)")
def acc_cells(files, ctx, spec):
    c = {}
    for f in files:
        for r in ok(load(f)):
            if plumbing(r["run_id"]) or int(r["context_length"]) != ctx or int(r["spec_steps"]) != spec:
                continue
            c.setdefault((int(r["seed"]), book(r["run_id"])), []).append(float(r["acceptance_rate"]))
    return {k: st.mean(v) for k, v in c.items()}
oa = acc_cells(OLD[(1048576, 4)], 1048576, 4)
na = acc_cells(NEW, 1048576, 4)
keys = sorted(set(oa) & set(na))
print("  n cells", len(keys))
for s in (42, 123, 456):
    ks = [k for k in keys if k[0] == s]
    om, nm = st.mean(oa[k] for k in ks), st.mean(na[k] for k in ks)
    print(f"  seed {s}: n={len(ks)} old {om:.4f} new {nm:.4f} change {100*(nm/om-1):+.2f}%")
om, nm = st.mean(oa[k] for k in keys), st.mean(na[k] for k in keys)
print(f"  pooled: old {om:.4f} new {nm:.4f} change {100*(nm/om-1):+.2f}%")
# old engine seed spread "23.0% of the mean on the matched books"
per_seed = {s: st.mean(oa[k] for k in keys if k[0] == s) for s in (42, 123, 456)}
print(f"  old per-seed means {per_seed}; (max-min)/mean = {100*(max(per_seed.values())-min(per_seed.values()))/st.mean(per_seed.values()):.2f}%")
# alternative: on books common to all three seeds (the 3 books seed 456 has)
common = sorted(set.intersection(*[{k[1] for k in keys if k[0] == s} for s in (42, 123, 456)]))
ps3 = {s: st.mean(oa[(s, b)] for b in common) for s in (42, 123, 456)}
print(f"  on common books {common}: per-seed {ps3} (max-min)/mean {100*(max(ps3.values())-min(ps3.values()))/st.mean(ps3.values()):.2f}%")
# per-book range across seeds / mean
rngs = [(b, (max(oa[(s, b)] for s in (42, 123, 456)) - min(oa[(s, b)] for s in (42, 123, 456))) / st.mean(oa[(s, b)] for s in (42, 123, 456))) for b in common]
print(f"  per-book seed range/mean: {[(b, round(100*x,1)) for b, x in rngs]} mean {100*st.mean(x for _, x in rngs):.2f}%")
cv = {s: None for s in ()}
allv = [oa[k] for k in keys]
print(f"  old stdev/mean over 13 cells {100*st.stdev(allv)/st.mean(allv):.2f}%")
# old buggy pool (with SA_SEED rows) for "4.033x" and "-10.4%"
bug = [R + "sampling_session/session_a_SAMP_1M_SPEC.csv", R + "seeds_ablations/session_a_SA_SEED_123_1M_SPEC.csv",
       R + "seeds_ablations/session_a_SA_SEED_456_1M_SPEC.csv"]
rows_bug = [r for f in bug for r in ok(load(f)) if int(r["spec_steps"]) == 4 and not plumbing(r["run_id"])]
nb = [r for f in NEW for r in ok(load(f)) if int(r["context_length"]) == 1048576 and int(r["spec_steps"]) == 4 and not plumbing(r["run_id"])]
print(f"  BUGGY pooled (rows labelled by stage): old acc {st.mean(float(r['acceptance_rate']) for r in rows_bug):.4f} "
      f"new acc {st.mean(float(r['acceptance_rate']) for r in nb):.4f} change "
      f"{100*(st.mean(float(r['acceptance_rate']) for r in nb)/st.mean(float(r['acceptance_rate']) for r in rows_bug)-1):+.2f}%; "
      f"decode speedup {st.mean(dec(r) for r in rows_bug)/st.mean(dec(r) for r in nb):.4f}")

print("=" * 70, "\n5. Rotary (F1)")
ROPE = {"native": ["sampling_session/session_a_ROPE_SAMP_NATIVE_SPEC.csv", "n5_extension/session_a_ROPE_EXT_SAMP_NATIVE_SPEC.csv"],
        "f16": ["sampling_session/session_a_ROPE_SAMP_F16_SPEC.csv", "n5_extension/session_a_ROPE_EXT_SAMP_F16_SPEC.csv"],
        "f32": ["sampling_session/session_a_ROPE_SAMP_F32_SPEC.csv", "n5_extension/session_a_ROPE_EXT_SAMP_F32_SPEC.csv"]}
acc = {}; tps = {}; pbk = {}
for a, fs in ROPE.items():
    rs = [r for f in fs for r in ok(load(R + f))]
    pbk[a] = {book(r["run_id"]): float(r["acceptance_rate"]) for r in rs}
    acc[a] = st.mean(pbk[a].values())
    tps[a] = st.mean(int(r["tokens_generated"]) / float(r["time_sec"]) for r in rs)
    print(f"  {a}: n={len(rs)} acc {acc[a]:.4f} tok/s {tps[a]:.4f}  recorded throughput_tps mean "
          f"{st.mean(float(r['throughput_tps']) for r in rs):.4f}")
for a in ("f16", "f32"):
    print(f"  {a} rel loss {100*(1-acc[a]/acc['native']):.2f}%  per-book drop holds: "
          f"{[(b, pbk['native'][b] > pbk[a][b], round(pbk[a][b]-pbk['native'][b],4)) for b in sorted(pbk['native'])]}")
print(f"  f16 vs f32 gap {acc['f16']-acc['f32']:.4f}, rel {100*(acc['f16']/acc['f32']-1):.2f}%")
# n=3 books gap
b3 = ["pg19_train_0", "pg19_train_1", "pg19_train_915"]
print(f"  n=3 gap f16-f32 {st.mean(pbk['f16'][b] for b in b3)-st.mean(pbk['f32'][b] for b in b3):.4f}")
gate = load(R + "sampling_session/session_a_rope_fix_gate.csv")
for g in gate:
    print(f"  gate {g['candidate']}: factor={g['rope_factor'] or '-'} ppl {g['ppl_continuation']} native {g['native_ppl_reference']} "
          f"ratio {g['ppl_ratio']} recomputed {float(g['ppl_continuation'])/float(g['native_ppl_reference']):.5f} "
          f"prompt {g['prompt_sha256'][:10]} cont {g['continuation_sha256'][:10]} rope_scaling {g['config_rope_scaling']}")

print("=" * 70, "\n6. Coherence (F3)")
q = {r["run_id"]: r for f in glob.glob(R + "sampling_session/session_a_QUALITY_FIX_*.csv") for r in ok(load(f))}
al = {r["doc_id"]: r for r in load(R + "sampling_session/alignment.csv")}
side = {}
for p in glob.glob(R + "sampling_session/tokens_slim/QUALITY_FIX_*.json"):
    b_ = json.load(open(p)); side[b_["run_id"]] = b_
for b in ("pg19_train_0", "pg19_train_1", "pg19_train_915"):
    a128, a1m = q[f"QUALITY_FIX_128K_{b}_s42"], q[f"QUALITY_FIX_1M_{b}_s42"]
    p128, p1m = float(a128["target_ppl"]), float(a1m["target_ppl"])
    s128 = side.get(f"QUALITY_FIX_128K_{b}_s42", {}).get("continuation_sha256")
    s1m = side.get(f"QUALITY_FIX_1M_{b}_s42", {}).get("continuation_sha256")
    print(f"  {b}: 128k {p128:.3f} 1M {p1m:.3f} move {100*(p1m/p128-1):+.2f}%  sidecar sha equal={s128==s1m} "
          f"alignment: offset {al[b]['offset_1M_vs_128k']} overlap {al[b]['overlap_pct_1M_vs_128k']} same {al[b]['same_continuation_1M_vs_128k']} "
          f"| alignment sha 1M==sidecar {al[b]['1M_continuation_sha256']==s1m} 128k==sidecar {al[b]['128k_continuation_sha256']==s128}")

print("=" * 70, "\n7. Ring bench")
rb = load(R + "decode_df/ring_bench/ringbench.csv")
ms = [float(r["ms_per_round"]) for r in rb]
sz = [float(r["size_mb_actual"]) for r in rb]
print(f"  {[(r['label'], r['size_mb_actual'], r['ms_per_round']) for r in rb]}")
print(f"  min {min(ms)} max {max(ms)} spread {(max(ms)/min(ms)-1)*100:.3f}%  payload range {max(sz)/min(sz):.0f}x "
      f"(bytes 279347200/4096 = {279347200/4096:.0f}x)")
rj = json.load(open(R + "decode_df/ring_bench/ringbench.json"))
print("  json keys:", list(rj)[:20])
for k in rj:
    if "skew" in k.lower() or "rank" in k.lower():
        print("   ", k, json.dumps(rj[k])[:400])

print("=" * 70, "\n8. vLLM")
v = ok(load(R + "vllm_run2/vllm_baseline.csv"))
for ctx in CT:
    vr = [r for r in v if int(r["context_length"]) == ctx]
    vdec = st.mean(float(r["decode_only_wall_s"]) / (int(r["output_tokens"]) - 1) for r in vr)
    vpre = st.mean(float(r["ttft_s"]) for r in vr)
    vtot = st.mean(float(r["end_to_end_wall_s"]) for r in vr)
    books_v = sorted(r["doc_id"] for r in vr)
    # RASD prefill: new-engine target-only rows on the same books (seed 42), plumbing dropped
    rs = [r for f in NEW for r in ok(load(f)) if int(r["context_length"]) == ctx and int(r["spec_steps"]) == 0
          and not plumbing(r["run_id"]) and book(r["run_id"]) in books_v]
    rs_spec = [r for f in NEW for r in ok(load(f)) if int(r["context_length"]) == ctx and int(r["spec_steps"]) == 4
               and not plumbing(r["run_id"])]
    pre = st.mean(float(r["ttft_ms"]) / 1000 for r in rs)
    pre_all = st.mean(float(r["ttft_ms"]) / 1000 for f in NEW + TOPUP for r in ok(load(f)) if int(r["context_length"]) == ctx and not plumbing(r["run_id"]))
    newt = st.mean(dec(r) for r in rs)
    oldt = head[(ctx, "target")]["old"]
    newspec = st.mean(dec(r) for r in rs_spec)
    print(f"  {CT[ctx]} vLLM books {books_v} T={[r['temperature'] for r in vr]} pre {vpre:.2f} dec {vdec:.5f} tot {vtot:.2f} | "
          f"RASD pre(same books, tgt) {pre:.2f} [all new rows {pre_all:.2f}] | tot/pre {vtot/pre:.3f} | "
          f"new/v {newt/vdec:.1f}x old/v {oldt/vdec:.1f}x spec/v {newspec/vdec:.1f}x  deficit removed {(1-(newt/vdec)/(oldt/vdec))*100:.1f}%")
    vo = ok(load(R + "vllm_run/vllm_baseline.csv"))
    vo_tot = st.mean(float(r["end_to_end_wall_s"]) for r in vo if int(r["context_length"]) == ctx)
    print(f"      vllm_run(old) e2e {vo_tot:.3f} vs run2 {vtot:.3f}: {100*(vo_tot/vtot-1):.2f}%  old ttft sample "
          f"{[r['ttft_s'] for r in vo if int(r['context_length'])==ctx][:1]}")

print("=" * 70, "\n9. Diagnostic arms")
arms = load(R + "decode_df/arms/df_arms.csv")
d = {(r["arm"], r["mode"]): r for r in arms}
for (a, m), r in d.items():
    print(f"  {a:11s} {m:6s} decode {float(r['decode_s_per_tok']):.4f} recomputed {dec(r):.4f} commit={r['git_commit']} "
          f"mode={r['ring_decode_mode']} L={r['ring_layers_patched']} quantize_target={r['quantize_target']} kv_quant={r['kv_quant']}")
t = lambda a: float(d[(a, "target")]["decode_s_per_tok"])
print(f"  ring2048/ring32768 {t('ring2048')/t('ring32768'):.4f}; ring32768/ring32768x {t('ring32768')/t('ring32768x'):.4f}; "
      f"ring2048/splitkv {t('ring2048')/t('splitkv'):.4f}; ring32768/splitkv {t('ring32768')/t('splitkv'):.4f}; "
      f"ring2048/ring32768x {t('ring2048')/t('ring32768x'):.4f}")

print("=" * 70, "\n10. Temperature arms")
for f in ("seeds_ablations/session_a_SA_TEMP_1_0_128K_SPEC.csv", "seeds_ablations/session_a_SA_TEMP_0_128K_SPEC.csv"):
    rs = ok(load(R + f)); print(f"  {f}: T={set(r['temperature'] for r in rs)} acc {st.mean(float(r['acceptance_rate']) for r in rs):.4f} n={len(rs)}")
rs = [r for f in OLD[(131072, 4)] for r in ok(load(f))]
print(f"  T=0.7 128k headline pool acc {st.mean(float(r['acceptance_rate']) for r in rs):.4f} n={len(rs)} books {sorted(book(r['run_id']) for r in rs)}")

json.dump(OUT, open("review_scratch/recompute_out.json", "w"), indent=1, default=str)

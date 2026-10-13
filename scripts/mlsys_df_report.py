#!/usr/bin/env python3
"""Turn the Part 1 evidence tree into the tables section F needs. Read-only, no GPU.

Usage:  python3 scripts/mlsys_df_report.py --dir <pulled evidence dir>

Reported:
  * the arm table at 128k (per arm x mode): tokens, time, ttft, decode s/tok, e2e tok/s,
    acceptance, peak memory, and the engine provenance each row recorded
  * the ratios against the in-session baseline, for target-only and spec separately
  * token-stream identity between arms: sha of the generated ids, first divergence index
    against the baseline, and the baseline's own median emitted top-2 gap at that index
    (a divergence at a near-tie is not a correctness signal; a divergence at a wide gap is)
  * the bare ring bench, against the two per-round figures the Phase C profile implies
  * the NCCL transport lines and the split-KV traceback, verbatim

The two per-round figures are the reason the bench exists:
  * 5.73 ms -- what the profile's SendRecv time costs PER ROUND (38.124 s over ~26 decode
    steps, divided by 256 rounds). Comparable to the bench's ms/round for the 266 MB payload.
  * 9.31 ms -- the context-INDEPENDENT term of the measured scaling fit (2.384 s / 256).
    If the bench's 266 MB round costs about this, the round cost no longer depends on
    payload and the bottleneck is serialization, not bandwidth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys

# From DECODE_DIAGNOSIS section C (the profiled 128k decode window).
PROFILE_MS_PER_ROUND_SENDRECV = 5.73
PROFILE_MS_PER_ROUND_INTERCEPT = 9.31


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def arms_section(d):
    rows = _rows(os.path.join(d, "arms", "df_arms.csv"))
    if not rows:
        return "## Arms\n\n(no arms CSV found)\n"
    out = ["## Arms at 128k (one model load, in-session baseline)\n"]
    hdr = ["arm", "mode", "tokens", "time_sec", "ttft_ms", "decode s/tok", "e2e tok/s",
           "accept", "peak_res MB", "engine recorded", "commit", "status"]
    tab = []
    for r in rows:
        if r.get("status") != "ok":
            tab.append([r.get("arm"), r.get("mode"), "-", "-", "-", "-", "-", "-", "-",
                        f"{r.get('ring_decode_mode')}/{r.get('kv_block_size')}/"
                        f"expand={r.get('expand_after_rotation')}",
                        (r.get("git_commit") or "")[:7], f"**{r.get('status')}**"])
            continue
        tab.append([
            r.get("arm"), r.get("mode"), r.get("tokens_generated"),
            f"{_f(r.get('time_sec')):.3f}", f"{_f(r.get('ttft_ms')):.0f}",
            f"{_f(r.get('decode_s_per_tok')):.4f}", f"{_f(r.get('e2e_tps')):.4f}",
            (f"{_f(r.get('acceptance_rate')):.4f}" if r.get("acceptance_rate") else "-"),
            f"{_f(r.get('peak_reserved_mb')):.0f}",
            f"{r.get('ring_decode_mode')}/{r.get('kv_block_size')}/"
            f"expand={r.get('expand_after_rotation')}",
            (r.get("git_commit") or "")[:7], r.get("status"),
        ])
    out.append(_md_table(hdr, tab))
    out.append("")
    out.append("Recorded `nccl_env`: " + (rows[0].get("nccl_env") or "(none set)") + "\n")

    # Ratios against the in-session baseline, per mode.
    base = {}
    for r in rows:
        if r.get("arm") == "ring2048" and r.get("status") == "ok":
            base[r["mode"]] = _f(r.get("decode_s_per_tok"))
    if base:
        out.append("### Ratios vs `ring2048` (target-only decode s/tok is the decision metric)\n")
        tab = []
        for r in rows:
            if r.get("status") != "ok" or r.get("arm") == "ring2048":
                continue
            m = r["mode"]
            b = base.get(m, float("nan"))
            v = _f(r.get("decode_s_per_tok"))
            tab.append([r.get("arm"), m, f"{b:.4f}", f"{v:.4f}",
                        f"{b / v:.4f}x" if v == v and v else "-"])
        out.append(_md_table(["arm", "mode", "ring2048 s/tok", "arm s/tok", "speedup"], tab))
        out.append("")

    # Token-stream identity.
    dig_path = os.path.join(d, "arms", "token_digests.json")
    if os.path.exists(dig_path):
        out.append("### Token streams (bit-identity claims)\n")
        digs = json.load(open(dig_path))
        tab = [[k, v] for k, v in sorted(digs.items())]
        out.append(_md_table(["arm|mode", "sha256[:16] of generated ids"], tab))
        out.append("")
        by_arm = {}
        for k, v in digs.items():
            arm, mode = k.split("|")
            by_arm.setdefault(mode, {})[arm] = v
        for mode, arms in sorted(by_arm.items()):
            b = arms.get("ring2048")
            if not b:
                continue
            for arm, digest in sorted(arms.items()):
                if arm == "ring2048":
                    continue
                same = "IDENTICAL" if digest == b else "**DIFFERS**"
                out.append(f"- `{mode}`: `{arm}` vs `ring2048` -> {same}")
        out.append("")
        out.append(_divergence_table(d, rows))
    return "\n".join(out) + "\n"


def _divergence_table(d, rows):
    """First divergence of each arm against the baseline, with the baseline's own top-2 gap.

    The gap is what makes a divergence readable: the merge's summation order differs by
    design, so a divergence at a position where the baseline's top-2 logits were within
    fp noise is expected; one at a wide gap is a correctness signal.
    """
    tokdir = os.path.join(d, "arms", "tokens")
    if not os.path.isdir(tokdir):
        return ""
    loaded = {}
    for r in rows:
        p = os.path.join(tokdir, f"{r['run_id']}.json")
        if r.get("status") == "ok" and os.path.exists(p):
            try:
                loaded[(r["arm"], r["mode"])] = json.load(open(p))
            except Exception:                                # noqa: BLE001
                pass
    base_gaps = {}
    for (arm, mode), blob in loaded.items():
        if arm != "ring2048":
            continue
        # `metrics.token_gaps` is the engine's own per-emitted-token top-2 margin and it IS
        # recorded for target-only rows (the spec path's `per_token_trace.emitted_gaps` is
        # the older source). Read the direct field first: a tool that silently sees an empty
        # list reports "no gaps around the divergence" rather than failing, which is how a
        # 0.0 near-tie can look like missing evidence.
        gaps = list(blob.get("token_gaps") or [])
        if not gaps:
            for rec in (blob.get("per_token_trace") or []):
                gaps.extend(rec.get("emitted_gaps") or [])
        if not gaps:
            for rec in ((blob.get("metrics") or {}).get("per_token_trace") or []):
                gaps.extend(rec.get("emitted_gaps") or [])
        gaps = [g for g in gaps if isinstance(g, (int, float))]
        if not gaps:
            gaps = [g for g in ((blob.get("metrics") or {}).get("token_gaps") or [])
                    if isinstance(g, (int, float))]
        if gaps:
            base_gaps[mode] = (statistics.median(gaps), min(gaps), max(gaps), len(gaps))

    out = ["### First divergence vs `ring2048`, and that row's own top-2 logit gap\n"]
    tab = []
    for (arm, mode), blob in sorted(loaded.items()):
        if arm == "ring2048":
            continue
        ref = loaded.get(("ring2048", mode))
        if not ref:
            continue
        a = blob.get("generated_token_ids") or []
        b = ref.get("generated_token_ids") or []
        div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if div is None:
            div = "none" if len(a) == len(b) else f"length {len(a)} vs {len(b)}"
        med = base_gaps.get(mode)
        gap = (f"median {med[0]:.4f} (min {med[1]:.4f})" if med else "n/a")
        # THE NUMBER THAT DECIDES THE GATE: the baseline's own margin at the position where
        # the arm diverged. Far below the row's median means a near-tie broke the other way.
        at_div = "n/a"
        if isinstance(div, int):
            bg = list(ref.get("token_gaps") or []) or \
                 list(((ref.get("metrics") or {}).get("token_gaps")) or [])
            if div < len(bg) and med:
                at_div = f"{bg[div]:.4f} ({'below' if bg[div] < med[0] else 'ABOVE'} median)"
        tab.append([arm, mode, div, len(a), len(b), at_div, gap])
    out.append(_md_table(["arm", "mode", "first divergence idx", "arm tokens",
                          "ring2048 tokens", "ring2048 gap AT the divergence",
                          "ring2048 gap distribution"], tab))
    out.append("")
    return "\n".join(out)


def bench_section(d):
    p = os.path.join(d, "ring_bench", "ringbench.csv")
    rows = _rows(p)
    if not rows:
        return "## Bare ring bench\n\n(no bench CSV found)\n"
    out = ["## The bare `batch_isend_irecv` ring, no model\n"]
    sorted_rows = sorted(rows, key=lambda r: _f(r.get("size_mb_actual"), 0))
    tab = []
    for r in sorted_rows:
        ms = _f(r.get("ms_per_round"))
        tot = ms * 256
        verdict = ""
        if tot == tot:
            if abs(tot - PROFILE_MS_PER_ROUND_INTERCEPT * 256) / (PROFILE_MS_PER_ROUND_INTERCEPT * 256) < 0.15:
                verdict = "~intercept (payload-independent)"
            elif tot < PROFILE_MS_PER_ROUND_INTERCEPT * 256:
                verdict = "faster than the engine's round"
        tab.append([r.get("label"), f"{_f(r.get('size_mb_actual')):.2f}",
                    r.get("seq_len"), r.get("heads"),
                    f"{ms:.4f}", f"{_f(r.get('ms_per_round_p90')):.4f}",
                    f"{_f(r.get('gb_per_s')):.2f}", f"{tot:.1f}", verdict])
    out.append(_md_table(["payload", "K+V MB", "seq", "heads", "ms/round", "p90 ms",
                          "GB/s", "ms / 256 rounds", "vs engine round"], tab))
    out.append("")
    out.append(f"For reference the engine's round, from the Phase C profile: "
               f"**{PROFILE_MS_PER_ROUND_SENDRECV} ms** of SendRecv per round and "
               f"**{PROFILE_MS_PER_ROUND_INTERCEPT} ms** per round as the context-independent "
               f"term of the measured scaling fit.\n")
    js = os.path.join(d, "ring_bench", "ringbench.json")
    if os.path.exists(js):
        skew = json.load(open(js)).get("skew") or []
        if skew:
            out.append("### Per-rank spread (is one rank a straggler?)\n")
            out.append(_md_table(["payload", "per-rank ms/round", "spread ms"],
                                 [[s["label"], s["per_rank_ms"], s["spread_ms"]] for s in skew]))
            out.append("")
    return "\n".join(out) + "\n"


def smoke_section(d):
    rows = _rows(os.path.join(d, "smoke", "df_smoke.csv"))
    out = ["## split-KV smoke at 8k (CUDA_LAUNCH_BLOCKING=1)\n"]
    for r in rows:
        out.append(f"- `{r.get('arm')}` {r.get('mode')}: status={r.get('status')} "
                   f"tokens={r.get('tokens_generated')} time={r.get('time_sec')} "
                   f"decode s/tok={r.get('decode_s_per_tok')}")
        if r.get("status") != "ok":
            out.append("\n```\n" + (r.get("error") or "")[:3000] + "\n```\n")
    if not rows:
        out.append("(no smoke CSV found)")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    d = a.dir
    parts = [arms_section(d), bench_section(d), smoke_section(d)]
    txt = "\n".join(parts)
    print(txt)
    outp = os.path.join(d, "REPORT_TABLES.md")
    with open(outp, "w") as fh:
        fh.write(txt)
    print(f"\n[df-report] wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

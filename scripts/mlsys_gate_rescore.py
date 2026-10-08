"""Re-score a completed gate run's nine rows under the CURRENT gate rule.

WHY THIS EXISTS
---------------
The 2026-10-08T14:23Z gate run measured every row correctly and then failed
three of its own positive controls, so the manifest aborted and the stage's CSV
was never pulled home. The only surviving record of the measurements is the
table the gate printed into `manifest.log`. The rule those numbers are judged
against has since changed (2026-10-08 plan revision: the generation shares are
relative to the paired baseline). This script re-applies the rule to the SAME
numbers, so the change can be shown to fix the calibration without spending a
GPU-hour and without re-measuring anything.

It deliberately imports `verdict` from the gate itself rather than restating the
rule: a re-score under a second, hand-written copy of the rule would prove
nothing about the production path.

WHAT IT CANNOT DO
-----------------
The log prints the verdict, not the reason every row got one, and it was
produced with the earlier absolute thresholds. Two consequences, both reported
in the output rather than smoothed over:

  * `early_eos` was computed with the withdrawn 16-token threshold. It is the
    same verdict under the new 10-token threshold for the rows in this log
    (N1's EOS is at token 5; no other row has one before 16), and that is
    checked, not assumed -- a row whose EOS position sits between 10 and 16
    would make the two rules disagree and this script refuses it.
  * the seed is not a column of the log. It is taken from the controls file, so
    the controls file passed here MUST be the revision that produced the log;
    a window is selected by (context, seed), so scoring an old log against a
    new controls file would pair rows that never shared a sample. `--controls`
    exists for exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.mlsys_coherence_gate import (  # noqa: E402
    CATASTROPHIC_EOS_TOKENS, PPL_TOLERANCE, GEN_SHARE_TOLERANCE, verdict)

# The printed table: name, ctx, anchor, ppl, ratio, blank, rep, eos, gate.
ROW_RE = re.compile(
    r"^\s+(?P<name>\S+)\s+(?P<ctx>\d+)\s+(?P<anchor>\S+)\s+"
    r"(?P<ppl>[0-9.]+)\s+(?P<ratio>[0-9.]+|)\s+(?P<blank>[0-9.]+)\s+"
    r"(?P<rep>[0-9.]+)\s+(?P<eos>YES|no)\s+(?P<gate>PASS|FAIL)\s*$")
EOS_AT_RE = re.compile(r"EOS at token (\d+)")


def parse_rows(log_text: str) -> list[dict]:
    """Every table row in the log, in order, with the reason lines attached."""
    lines = log_text.splitlines()
    rows, reason_for = [], None
    for i, line in enumerate(lines):
        m = ROW_RE.match(line)
        if m:
            rows.append(m.groupdict())
            # `/^\s+-> /` is the gate's indented continuation line: the verbatim
            # reasons, which is where the EOS position lives.
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            reason_for = nxt.strip()[3:].strip() if nxt.strip().startswith("->") \
                else ""
            rows[-1]["reasons"] = reason_for
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--controls", default=str(REPO / "configs"
                                            / "mlsys_gate_controls.json"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.controls).read_text())
    declared = {c["name"]: c for c in spec["candidates"]}
    log_rows = parse_rows(args.log.read_text(errors="replace"))
    if not log_rows:
        raise SystemExit(f"no gate table found in {args.log}; refusing to "
                         f"re-score nothing")

    rows = []
    for lr in log_rows:
        d = declared.get(lr["name"])
        if d is None:
            raise SystemExit(
                f"row {lr['name']} is in the log but not in {args.controls}; "
                f"the pairing would be invented. Pass the controls revision "
                f"that produced this log.")
        eos_at = EOS_AT_RE.search(lr["reasons"] or "")
        eos_at = int(eos_at.group(1)) if eos_at else None
        # The log's `early_eos` came from the withdrawn 16-token threshold. The
        # only thing that matters here is whether the NEW threshold agrees, so
        # check it: YES with an EOS at >= 10 would be a disagreement.
        early_new = lr["eos"] == "YES" and (eos_at is None
                                           or eos_at < CATASTROPHIC_EOS_TOKENS)
        rows.append({
            "candidate": lr["name"],
            "target_model_name": d["target_model_name"],
            "context_length": int(lr["ctx"]),
            "seed": d.get("seed", 42),
            "native_baseline": bool(d.get("native_baseline", False)),
            "role": d.get("role", ""),
            "expect": d.get("expect", ""),
            "status": "ok",
            "ppl_continuation": float(lr["ppl"]),
            "gen_blank_share": float(lr["blank"]),
            "gen_repeat_share": float(lr["rep"]),
            "early_eos": early_new,
            "eos_at": eos_at if eos_at is not None else "",
            "effective_rope_matches_intent": lr["anchor"] == "declared",
            "effective_rope_match": lr["anchor"],
            "old_gate_pass": lr["gate"] == "PASS",
            "old_reasons": lr["reasons"],
            "log_early_eos_agrees": (lr["eos"] != "YES") or early_new,
        })

    baseline_rows = [r for r in rows
                     if r["native_baseline"] and r["ppl_continuation"]]

    def baseline_for(r):
        want_ctx = r.get("reference_context") or r["context_length"]
        same = [b for b in baseline_rows
                if b["context_length"] == int(want_ctx)
                and b["target_model_name"] == r["target_model_name"]]
        if not same:
            return None
        same_seed = [b for b in same if b.get("seed") == r.get("seed")]
        return (same_seed or same)[0]

    out_lines = []
    out_lines.append(f"re-score of {args.log}")
    out_lines.append(f"controls file: {args.controls}")
    out_lines.append(
        f"rule: ppl <= {PPL_TOLERANCE}x baseline; blank/repeat <= "
        f"{GEN_SHARE_TOLERANCE}x the baseline's share; absolute ceilings only "
        f"for degeneration (blank > 0.90, EOS before token "
        f"{CATASTROPHIC_EOS_TOKENS})")
    out_lines.append("")
    hdr = (f"  {'candidate':<28} {'expect':<5} {'old':<5} {'new':<5} "
           f"{'ppl_ratio':>9} {'blank':>7} {'b/base':>8} {'rep':>7} "
           f"{'r/base':>7} {'eos':>4}  reason")
    out_lines.append(hdr)
    disagreements = []
    for r in rows:
        b = baseline_for(r)
        v = verdict(r, b)
        new_pass = v["gate_pass"]
        want = (r["expect"] == "pass")
        mark = "ok " if new_pass == want else "BAD"
        if new_pass != want:
            disagreements.append(r["candidate"])
        if not r["log_early_eos_agrees"]:
            disagreements.append(f"{r['candidate']} (early_eos threshold)")
        out_lines.append(
            f"{mark} {r['candidate']:<28} {r['expect']:<5} "
            f"{'PASS' if r['old_gate_pass'] else 'FAIL':<5} "
            f"{'PASS' if new_pass else 'FAIL':<5} "
            f"{v['ppl_ratio']!s:>9} {r['gen_blank_share']!s:>7} "
            f"{v.get('baseline_blank_share')!s:>8} "
            f"{r['gen_repeat_share']!s:>7} {v.get('repeat_share_ratio')!s:>7} "
            f"{'YES' if r['early_eos'] else 'no':>4}  "
            f"{v['gate_reason'][:96]}")
    out_lines.append("")
    for r in rows:
        if r["old_reasons"]:
            out_lines.append(f"  old reason | {r['candidate']}: "
                             f"{r['old_reasons'][:150]}")
    out_lines.append("")
    n_pos = sum(1 for r in rows if r["expect"] == "pass")
    n_neg = sum(1 for r in rows if r["expect"] == "fail")
    out_lines.append(f"  {n_pos} positives declared pass, {n_neg} negatives "
                     f"declared fail")
    if disagreements:
        out_lines.append(f"  CALIBRATION WOULD FAIL: {disagreements}")
    else:
        out_lines.append("  CALIBRATION PASSES: every control has its "
                         "declared verdict under the new rule")

    text = "\n".join(out_lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())

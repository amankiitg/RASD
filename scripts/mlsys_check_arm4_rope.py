#!/usr/bin/env python
"""ARM4 pre-flight: verify the extrapolation ladder WITHOUT a GPU.

Run this before spending anything. It is free, CPU-only, and needs only the
HF token (it downloads config.json, not weights).

What it checks, per ARM4 rung:

  1. The config file expands to the expected run ids and contexts
     (ARM4_llama3_yarn_<ctx>_s<seed>), with a matched spec_steps=0 baseline.
  2. The EXACT ``rope_scaling`` dict the engine will apply to the target,
     computed by the real ``_build_hf_config`` code path (not a
     reimplementation), together with the effective factor and its base.
  3. The ladder assertions:
       128k -> factor 1     (rope_type="none"; native block untouched)
       256k -> factor 2     base 131072
       512k -> factor 4     base 131072
     A wrong rope config would silently invalidate the entire Phase-A result,
     so any failure here is fatal: exit 1 and DO NOT LAUNCH.
  4. The draft window cap is 4096 on every rung (held fixed by design).
  5. max_new_tokens == 128 on every ARM4 cell.

Usage:
    python scripts/mlsys_check_arm4_rope.py
    python scripts/mlsys_check_arm4_rope.py --target meta-llama/Llama-3.1-8B

Exit: 0 = ladder verified, 1 = FAILED (do not launch), 2 = inconclusive
      (e.g. no HF access to the target config).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# (config file, expected context, expected factor, rope_type, expectation)
# expectation: "native" = shipped block preserved; "yarn" = yarn over base;
#              "llama3" = Meta's own dict with only factor overridden
LADDER = [
    ("configs/mlsys_arm4_f1_128k.yml",       130944, 1,  "yarn",   "native"),
    ("configs/mlsys_arm4_f2_128k.yml",       130944, 2,  "yarn",   "yarn"),
    ("configs/mlsys_arm4_f4_128k.yml",       130944, 4,  "yarn",   "yarn"),
    ("configs/mlsys_arm4_f2_256k.yml",       262016, 2,  "yarn",   "yarn"),
    ("configs/mlsys_arm4_f4_512k.yml",       524160, 4,  "yarn",   "yarn"),
    ("configs/mlsys_arm4_llama3f16_128k.yml", 130944, 16, "llama3", "llama3"),
    ("configs/mlsys_arm4_llama3f32_128k.yml", 130944, 32, "llama3", "llama3"),
]
NATIVE_WINDOW = 131072      # Llama-3.1-8B max_position_embeddings
EXPECTED_DRAFT_CAP = 4096
EXPECTED_MAX_NEW = 128      # B3: exactly this many tokens, EOS ignored


def header(msg: str) -> None:
    print(f"\n{'=' * 74}\n{msg}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="meta-llama/Llama-3.1-8B")
    args = ap.parse_args()

    from transformers import AutoConfig
    from src.models.rasd_inference import RASDInference
    from run_experiment import build_run_configs, load_config

    header("ARM4 rope ladder — CPU pre-flight (no GPU, no weights)")

    # The native window is read from the real model, not assumed.
    try:
        native_cfg = AutoConfig.from_pretrained(args.target)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR loading {args.target} config: {e}")
        print("  (gated model — is HF_TOKEN set and accepted?)")
        return 2
    native_max = int(native_cfg.max_position_embeddings)
    native_rope = getattr(native_cfg, "rope_scaling", None)
    print(f"  target            : {args.target}")
    print(f"  native window     : {native_max}")
    print(f"  native rope block : {native_rope}")
    if native_max != NATIVE_WINDOW:
        print(f"  NOTE: native window {native_max} != assumed {NATIVE_WINDOW}; "
              f"assertions below use the REAL value.")

    # Bypass __init__ — we only need the pure config-building method.
    engine = object.__new__(RASDInference)
    all_ok = True

    for cfg_path, ctx, want_factor, want_type, expect in LADDER:
        header(f"{cfg_path}  (ctx={ctx})")
        p = REPO / cfg_path
        if not p.exists():
            print(f"  [FAIL] config missing: {cfg_path}")
            all_ok = False
            continue

        cfg = load_config(str(p))
        runs = build_run_configs(cfg, groups=None, debug=False)
        spec = [r for r in runs if r.get("spec_steps", 0) != 0]
        tgt = [r for r in runs if r.get("spec_steps", 0) == 0]
        ids = sorted({r["run_id"] for r in spec})

        print(f"  runs expanded     : {len(runs)} "
              f"({len(spec)} spec, {len(tgt)} target-only)")
        print(f"  spec run_ids      : {ids}")
        print(f"  contexts          : {sorted({r['context_length'] for r in runs})}")

        # Per-rung flag: accumulating straight into all_ok would make the
        # "[PASS]" lines below lie about THIS rung as soon as an earlier rung
        # had failed.
        rung_ok = True

        level_ids = sorted({r["level_id"] for r in spec})
        base_id = level_ids[0] if level_ids else ""
        # (1) shape
        want_ids = [f"{base_id}_s{s}" for s in (42, 123, 456)]
        # Compare as SETS: `sorted()` is lexical, so the seed suffixes come
        # out as _s123, _s42, _s456 and an order-sensitive compare would
        # reject a perfectly good config.
        if set(ids) != set(want_ids):
            print(f"  [FAIL] run_ids != expected {want_ids}")
            rung_ok = False
        else:
            print("  [PASS] run_ids and 3 seeds match the expected pattern")
        declares_baseline = "ARM4_TARGET_ONLY" in cfg
        if declares_baseline and not tgt:
            print("  [FAIL] config declares ARM4_TARGET_ONLY but no rows")
            rung_ok = False
        elif tgt:
            print(f"  [PASS] matched target-only baseline present ({len(tgt)} run(s))")
        else:
            print("  [OK] spec-only cell (baseline optional per addendum)")
        if {r["context_length"] for r in runs} != {ctx}:
            print(f"  [FAIL] context mismatch (want {ctx})")
            rung_ok = False

        # (4)/(5) held-fixed knobs
        knobs_ok = True
        for r in runs:
            if r.get("draft_window_cap") != EXPECTED_DRAFT_CAP:
                print(f"  [FAIL] {r['run_id']}: draft_window_cap="
                      f"{r.get('draft_window_cap')} != {EXPECTED_DRAFT_CAP}")
                knobs_ok = False
            if r.get("max_new_tokens") != EXPECTED_MAX_NEW:
                print(f"  [FAIL] {r['run_id']}: max_new_tokens="
                      f"{r.get('max_new_tokens')} != {EXPECTED_MAX_NEW}")
                knobs_ok = False
        if knobs_ok:
            print(f"  [PASS] draft_window_cap={EXPECTED_DRAFT_CAP} and "
                  f"max_new_tokens={EXPECTED_MAX_NEW} on every rung")
        rung_ok &= knobs_ok

        # (2)/(3) the rope dict the engine will actually apply
        hf = engine._build_hf_config(
            args.target, None, ctx, "target",
            apply_rope_scaling=True, rope_type=want_type,
            rope_factor=want_factor,
        )
        rs = getattr(hf, "rope_scaling", None)
        print(f"  applied rope_scaling : {rs}")
        print(f"  applied max_pos      : {hf.max_position_embeddings}")

        if expect == "native":
            if rs != native_rope:
                print("  [FAIL] rope_type='none' modified the model's native "
                      "rope block — the 128k rung would not replicate Arm2")
                rung_ok = False
            else:
                print(f"  [PASS] factor 1: native block untouched "
                      f"(= Arm2, in-distribution anchor)")
        elif expect == "llama3":
            if not isinstance(rs, dict) or rs.get("rope_type") != "llama3":
                print(f"  [FAIL] expected Meta's llama3 dict, got {rs}")
                rung_ok = False
            elif rs.get("factor") != float(want_factor):
                print(f"  [FAIL] llama3 factor {rs.get('factor')} != {want_factor}")
                rung_ok = False
            else:
                print(f"  [PASS] llama3 mechanism kept, factor == {want_factor}")
        elif not isinstance(rs, dict) or rs.get("type") != "yarn":
            # No `continue` here: it would skip the accumulator below and
            # let a broken rung still report an overall PASS.
            print(f"  [FAIL] expected a yarn rope_scaling dict, got {rs}")
            rung_ok = False
        else:
            got_f = rs.get("factor")
            got_base = rs.get("original_max_position_embeddings")
            if got_f != want_factor:
                print(f"  [FAIL] factor {got_f} != {want_factor}")
                rung_ok = False
            else:
                print(f"  [PASS] factor == {want_factor}")
            if got_base != native_max:
                print(f"  [FAIL] base {got_base} != native window {native_max} "
                      f"(a hardcoded 4096 would silently mis-scale this cell)")
                rung_ok = False
            else:
                print(f"  [PASS] base == {native_max} (model-native, not 4096)")
            # The window must COVER the prompt (B2). For the matched-context
            # cells ctx = native - 128, so the window correctly stays at the
            # native 131072 rather than being lowered to ctx.
            if hf.max_position_embeddings < ctx:
                print(f"  [FAIL] window {hf.max_position_embeddings} < ctx {ctx}")
                rung_ok = False
            else:
                print(f"  [PASS] window {hf.max_position_embeddings} covers ctx {ctx}")

        all_ok &= rung_ok

    header("ARM4 rope ladder verdict")
    if all_ok:
        print("  PASS — the target's rope config is correct at every rung.")
        print("         128k: native (factor 1, replicates Arm2)")
        print("         256k: YaRN factor 2 over base 131072")
        print("         512k: YaRN factor 4 over base 131072")
        return 0
    print("  FAIL — the ladder is NOT safe to run. Do not launch.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Drop config levels whose rope configuration the gate did NOT pass.

`scripts/mlsys_manifest.sh` declared `require_gate_verdict: pass` for every
`>128k` stage but never enforced it, so a gate that failed every candidate still
let every stage run. This script is the enforcement: it reads the gate verdicts
and writes a filtered copy of the stage config containing only the levels whose
configuration passed.

Matching rule: a level's group name with a leading `gated_` removed must appear
as a gate candidate name (case-insensitive). A level that matches NO gate
candidate is treated as NOT passed, because the safe default for "we could not
find a verdict" is to refuse to run rather than to assume it is fine.

Usage:
    python scripts/mlsys_gate_filter.py --gate results/mlsys/coherence_gate.csv \
        --config configs/mlsys_natural_gated.yml --out /tmp/filtered.yml
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

NON_GROUP = ("defaults", "canary")


def gate_verdicts(gate_csv: Path) -> dict:
    out = {}
    for row in csv.DictReader(gate_csv.open()):
        name = str(row.get("candidate", "")).strip().lower()
        if name:
            out[name] = str(row.get("gate_pass", "")).strip() == "True"
    return out


def normalise(group_name: str) -> str:
    """`GATED_llama3_f16_256k` -> `llama3_f16_256k`."""
    g = group_name.strip().lower()
    for pre in ("gated_", "gate_"):
        if g.startswith(pre):
            g = g[len(pre):]
    return g


def filter_config(config_path: Path, verdicts: dict):
    cfg = yaml.safe_load(config_path.read_text())
    kept, dropped = {}, []
    for key, val in cfg.items():
        if key in NON_GROUP:
            kept[key] = val
            continue
        norm = normalise(str(key))
        # Also try the level ids: a group may hold several levels and the gate
        # names the configuration, which the group name carries.
        passed = verdicts.get(norm)
        if passed is None:
            alts = [k for k in verdicts if k in norm or norm in k]
            passed = any(verdicts[a] for a in alts) if alts else None
        if passed is True:
            kept[key] = val
        else:
            dropped.append({
                "group": key,
                "reason": ("no gate verdict matched this configuration"
                           if passed is None else "gate verdict was FAIL"),
            })
    return kept, dropped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-empty", action="store_true",
                   help="Exit 0 even when nothing passed (the runner decides)")
    args = p.parse_args()

    gate_csv = Path(args.gate)
    if not gate_csv.exists() or not gate_csv.stat().st_size:
        print(f"REFUSING: no gate verdicts at {gate_csv}; nothing may run above 128k")
        return 2
    verdicts = gate_verdicts(gate_csv)
    kept, dropped = filter_config(Path(args.config), verdicts)

    for d in dropped:
        print(f"  DROPPED {d['group']}: {d['reason']}")
    groups = [k for k in kept if k not in NON_GROUP]
    print(f"  kept {len(groups)} of {len(groups) + len(dropped)} groups from "
          f"{args.config}")

    if not groups:
        print("  no configuration passed the gate; nothing to run")
        if not args.allow_empty:
            return 1
    Path(args.out).write_text(yaml.safe_dump(kept, sort_keys=False, width=1000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

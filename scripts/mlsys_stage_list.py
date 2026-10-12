#!/usr/bin/env python3
"""The stage list of a session config: one implementation, two callers.

`mlsys_session_a_remote.sh` derives the stages it will run from the config file's key
ORDER, and the parameter contract needs the driver to know the same list so it can
assert that the pod is running the stages it intended. Two copies of that rule -- one
in the pod script's inline python, one in the driver -- is exactly how a campaign
starts disagreeing with itself about what it ran, so both now call this.

The rule: every top-level key except `defaults` and `canary`, in file order. `canary`
is reserved; `defaults` is the shared block, not a stage. A metadata key added as
YAML is therefore executed as a stage, which is why the configs carry their notes in
COMMENTS rather than in a `_comment` key.

Usage:  python3 scripts/mlsys_stage_list.py <config.yml> [--space]
Exit:   0 with the stage names on stdout (one per line, or space-separated);
        3 with a message on stderr when the config cannot be read, so a caller can
        tell "no stages" apart from "no file".
"""
from __future__ import annotations

import sys


def stage_list(config_path: str) -> list[str]:
    import yaml
    cfg = yaml.safe_load(open(config_path))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{config_path}: not a mapping at the top level")
    return [k for k in cfg if k not in ("defaults", "canary")]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 3
    path = sys.argv[1]
    space = "--space" in sys.argv[2:]
    try:
        stages = stage_list(path)
    except FileNotFoundError:
        print(f"stage_list: no such config: {path}", file=sys.stderr)
        return 3
    except Exception as exc:                                   # noqa: BLE001
        print(f"stage_list: cannot read {path}: {exc}", file=sys.stderr)
        return 3
    print(" ".join(stages) if space else "\n".join(stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

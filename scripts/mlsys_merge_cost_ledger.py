"""Append a pod's cost ledger into the local one, without losing either side.

WHY THIS IS NOT A COPY
----------------------
`collect_incident` pulls a failed run's `results/mlsys/` home before it
terminates the instance. Every other file there can simply be copied because the
pull lands in a fresh incident directory. The cost ledger is the exception:
`results/mlsys/gpu_hours.csv` is the campaign's running total locally, and the
pod's copy is a SUPERSET of it (the pod appends a row as it spends), so
overwriting in either direction loses rows. Before this existed, an aborted 8x
run lost its spend entirely: the guard's `spend()` read the local file, the
stage's row only ever reached the pod, and the instance was terminated seconds
later.

MERGE RULE
----------
A pod row is appended only if the local file does not already hold a row with the
same bytes, counting MULTIPLICITY: the merge is a multiset union of the body
lines, so two genuinely distinct runs that happen to produce byte-identical rows
are both kept, and running the merge twice is a no-op the second time. Order is
preserved -- local rows first, then the pod's new rows in the order the pod wrote
them.

A header mismatch refuses rather than appending rows into a file whose columns it
does not know. The caller keeps the pod's copy in the incident directory, so
refusing loses nothing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse(text: str) -> tuple:
    """(header, body lines), ignoring blank lines."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return "", []
    return lines[0], lines[1:]


def merge(local_text: str, pod_text: str) -> dict:
    """The merged ledger and what happened.

    `status` is one of:
      created         local had no ledger at all; the pod's file is taken as is
      no_pod_rows     the pod's ledger is empty; local is returned unchanged
      header_mismatch the two files disagree about their columns; refuse
      merged          `appended` rows were added
      unchanged       the pod held nothing the local file did not already have
    """
    if not local_text.strip():
        if not pod_text.strip():
            return {"status": "no_pod_rows", "text": "", "appended": 0}
        _, prows = parse(pod_text)
        return {"status": "created", "text": pod_text, "appended": len(prows)}

    local_header, local_rows = parse(local_text)
    pod_header, pod_rows = parse(pod_text)
    if not pod_rows:
        return {"status": "no_pod_rows", "text": local_text, "appended": 0}
    if local_header != pod_header:
        return {"status": "header_mismatch", "text": local_text, "appended": 0,
                "local_header": local_header, "pod_header": pod_header}

    have = Counter(local_rows)
    seen = Counter()
    add = []
    for line in pod_rows:
        seen[line] += 1
        if seen[line] > have[line]:
            add.append(line)

    lines = [local_header] + local_rows + add
    return {"status": "merged" if add else "unchanged",
            "text": "\n".join(lines) + "\n",
            "appended": len(add),
            "local_rows": len(local_rows),
            "total_rows": len(local_rows) + len(add)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, type=Path)
    ap.add_argument("--pod", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="write here instead of in place (default: --local)")
    args = ap.parse_args()

    local_text = args.local.read_text() if args.local.exists() else ""
    pod_text = args.pod.read_text() if args.pod.exists() else ""
    result = merge(local_text, pod_text)
    out = args.out or args.local
    if result["status"] in ("created", "merged") and result["text"]:
        out.write_text(result["text"])
    summary = {k: v for k, v in result.items() if k != "text"}
    print(json.dumps(summary, sort_keys=True))
    return 2 if result["status"] == "header_mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())

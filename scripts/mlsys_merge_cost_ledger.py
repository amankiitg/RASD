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

LINE ENDINGS ARE PART OF THE DATA
---------------------------------
The ledger committed in this repo uses CRLF; the pod appends with `echo` and
`awk`, which write LF. Reading with universal newlines and writing the result
back therefore rewrote every line of a file the merge had not been asked to
change -- a 7-line ledger showed as 7 deletions and 7 insertions. So the text is
read with `newline=""` and each row keeps its own terminator: rows that are kept
are re-emitted byte for byte, and appended rows are written in the file's own
convention. Only the comparison ignores the terminator.

A header mismatch refuses rather than appending rows into a file whose columns it
does not know. The caller keeps the pod's copy in the incident directory, so
refusing loses nothing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _read_raw(path: Path) -> str:
    """Read without newline translation: the bytes decide the terminator."""
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def _lines(text: str) -> list:
    """The file's rows, each keeping its own line terminator."""
    if not text.strip():
        return []
    rows = text.split("\n")
    if rows and rows[-1] == "":
        rows.pop()                      # the piece after the final newline
    return [r for r in rows if r.strip()]


def _bare(row: str) -> str:
    """A row without its terminator, for comparison and de-duplication."""
    return row[:-1] if row.endswith("\r") else row


def _terminator(rows: list) -> str:
    """The convention this file uses, taken from its first row."""
    return "\r" if rows and rows[0].endswith("\r") else ""


def parse(text: str) -> tuple:
    """(header, body lines), terminators intact."""
    rows = _lines(text)
    if not rows:
        return "", []
    return rows[0], rows[1:]


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
        return {"status": "created", "text": pod_text,
                "appended": len(parse(pod_text)[1])}

    local_rows = _lines(local_text)
    pod_rows = _lines(pod_text)
    if not pod_rows:
        return {"status": "no_pod_rows", "text": local_text, "appended": 0}
    if _bare(local_rows[0]) != _bare(pod_rows[0]):
        return {"status": "header_mismatch", "text": local_text, "appended": 0,
                "local_header": _bare(local_rows[0]),
                "pod_header": _bare(pod_rows[0])}

    cr = _terminator(local_rows)
    have = Counter(_bare(r) for r in local_rows[1:])
    seen = Counter()
    add = []
    for row in pod_rows[1:]:
        bare = _bare(row)
        seen[bare] += 1
        if seen[bare] > have[bare]:
            add.append(bare + cr)       # written in the LOCAL file's convention

    out = local_rows + add
    return {"status": "merged" if add else "unchanged",
            "text": "\n".join(out) + "\n",
            "appended": len(add),
            "local_rows": len(local_rows) - 1,
            "total_rows": len(local_rows) - 1 + len(add)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, type=Path)
    ap.add_argument("--pod", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="write here instead of in place (default: --local)")
    args = ap.parse_args()

    local_text = _read_raw(args.local)
    pod_text = _read_raw(args.pod)
    result = merge(local_text, pod_text)
    out = args.out or args.local
    if result["status"] in ("created", "merged") and result["text"]:
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write(result["text"])
    summary = {k: v for k, v in result.items() if k != "text"}
    print(json.dumps(summary, sort_keys=True))
    return 2 if result["status"] == "header_mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())

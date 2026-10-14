#!/usr/bin/env bash
# Merge a VERIFIED results pull into the repository's results directory.
#
# Extracted from the watcher so the rule is testable. The rule:
#
#   * copy into results/ only if the pull is complete AND every file's sha256
#     matches the remote's own manifest;
#   * on any doubt, merge NOTHING and exit non-zero, leaving the staged copy in
#     place for inspection.
#
# It was extracted because the previous inline version had the merge OUTSIDE the
# guard: `PULL_FAILED` only chose the wording of a message while a partial pull
# was copied into results/ anyway. That is the worst shape for this bug, because
# a merged short CSV is indistinguishable from a short run -- the failure is
# discovered during analysis, if at all. A rule that cannot be exercised without
# a GPU and an SSH session is a rule that is not enforced.
#
# Usage:
#   mlsys_pull_merge.sh --stage-dir STAGED_PULL --remote-sha FILE
#                       [--dest results/mlsys] [--local-sha FILE] [--log FILE]
#
# Exit codes:
#   0  merged (all files verified)
#   1  the staged pull is missing or empty: nothing to merge
#   2  the remote manifest is missing or empty: verification impossible
#   3  a sha256 mismatch: nothing merged
set -uo pipefail

STAGE=""; REMOTE_SHA=""; DEST="results/mlsys"; LOCAL_SHA=""; LOG=/dev/null
while [ $# -gt 0 ]; do
  case "$1" in
    --stage-dir)  STAGE=${2:-}; shift 2 ;;
    --remote-sha) REMOTE_SHA=${2:-}; shift 2 ;;
    --dest)       DEST=${2:-}; shift 2 ;;
    --local-sha)  LOCAL_SHA=${2:-}; shift 2 ;;
    --log)        LOG=${2:-}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
[ -n "$STAGE" ] && [ -n "$REMOTE_SHA" ] || {
  echo "usage: mlsys_pull_merge.sh --stage-dir DIR --remote-sha FILE" >&2
  exit 64
}
say() { printf '%s\n' "$*"; printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

# 1. Is there a pull at all?
if [ ! -d "$STAGE" ] || [ -z "$(find "$STAGE" -type f -print -quit 2>/dev/null)" ]; then
  say "!!! NOT MERGED: no files in the staged pull at $STAGE; rsync must have"\
      "failed or copied nothing. results/ is untouched."
  exit 1
fi

# 2. Is there something to verify against? An absent remote manifest means the
#    pull is unverifiable, and "unverifiable" is not "verified".
if [ ! -s "$REMOTE_SHA" ]; then
  say "!!! NOT MERGED: no remote sha256 manifest at $REMOTE_SHA, so the pull"\
      "cannot be verified. results/ is untouched."
  exit 2
fi

# 3. Recompute the staged side. `sort -z` on both sides so the two manifests are
#    comparable line-for-line; the paths are relative to the same root, which is
#    what makes a straight diff meaningful.
if [ -z "$LOCAL_SHA" ]; then
  LOCAL_SHA=$(mktemp -t mlsys_pull_local.XXXXXX)
fi
( cd "$STAGE" && find . -type f -print0 | sort -z | xargs -0 shasum -a 256 ) \
  >"$LOCAL_SHA" 2>>"$LOG"
if [ ! -s "$LOCAL_SHA" ]; then
  say "!!! NOT MERGED: could not compute the staged sha256 manifest."
  exit 2
fi

if ! diff -u <(sort "$REMOTE_SHA") <(sort "$LOCAL_SHA") >>"$LOG" 2>&1; then
  say "!!! NOT MERGED: sha256 MISMATCH between the remote and the staged pull."
  say "!!!   remote files: $(wc -l < "$REMOTE_SHA" | tr -d ' ')  staged files: $(wc -l < "$LOCAL_SHA" | tr -d ' ')"
  diff <(sort "$REMOTE_SHA") <(sort "$LOCAL_SHA") 2>&1 | head -20 | while IFS= read -r l; do
    say "!!!   $l"
  done
  say "!!! staged copy kept at $STAGE for inspection"
  exit 3
fi

# 4. Verified. Merge additively: never --delete, because results/ holds runs from
#    earlier stages and an instance that only wrote the newest stage would
#    otherwise wipe them.
n=0
while IFS= read -r -d '' f; do
  rel=${f#"$STAGE"/}
  mkdir -p "$DEST/$(dirname "$rel")"
  cp -p "$f" "$DEST/$rel"
  n=$((n + 1))
done < <(find "$STAGE" -type f -print0)
say "  merged $n files into $DEST (additive; nothing deleted)"
exit 0

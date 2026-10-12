#!/usr/bin/env bash
# CHOOSE AN AVAILABLE 1x GPU TYPE, THEN HAND THE RUN TO THE SESSION DRIVER.
#
# WHY THIS EXISTS. `scripts/mlsys_session_a.sh` polls ONE instance type, because that is
# what a paid run must do -- "the default is the only type the main session may use, and
# nothing else may override it". A 1x REHEARSAL tests plumbing rather than timing, so any
# 1x card will do, and today the A100 SXM4 1x pool has been empty for half an hour while
# the rehearsal's deadline counts down. The driver cannot be edited to say that (a running
# session is executing it), so the preference list lives here and the chosen type is
# pinned into the driver's own SESSION_TYPE, where the driver's rule is still enforced.
#
# The type is recorded, with the rate the API reports for it, because a rehearsal run on a
# different card is a different experiment in one respect: which GPU the plumbing was
# exercised on.
#
# Usage (the driver's own variables pass straight through):
#   TIMER_MIN=60 SESSION_TAG=... SESSION_CFG=... bash scripts/mlsys_rehearsal_pick_1x.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 3
LOG=${PICK_LOG:-/tmp/mlsys_pick_1x.log}
TYPES=${REHEARSAL_TYPES:-"gpu_1x_a100_sxm4 gpu_1x_a100 gpu_1x_h100_pcie gpu_1x_a6000 gpu_1x_a10"}
MAX_RATE=${REHEARSAL_MAX_RATE:-3.00}     # <= $3 for a 60-minute timer, per the plan
DEADLINE_S=${PICK_DEADLINE_S:-10800}     # keep re-polling for up to 3 h
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

KEY=$(grep -E '^[[:space:]]*LAMBDA_API_KEY[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
[ -n "${KEY:-}" ] || { say "FATAL: no API key in runpod_creds.md"; exit 2; }

say "preference order: $TYPES (rate ceiling \$$MAX_RATE/hr)"
END=$(( $(date -u +%s) + DEADLINE_S ))
PICK_PY=${PICK_PY:-python3}
CHOSEN=""; RATE=""; REGION=""
PICKTMP=$(mktemp)
while :; do
  AVAIL=$(curl -sS --max-time 60 -u "$KEY:" "https://cloud.lambda.ai/api/v1/instance-types" 2>/dev/null)
  printf '%s' "$AVAIL" > "$PICKTMP"
  # THE JSON GOES IN A FILE, NOT ON STDIN. The first version piped it into a python whose
  # SCRIPT came from a heredoc, so the heredoc consumed stdin and `json.load` read EOF:
  # the picker then selected nothing every time and -- worse -- exported an EMPTY
  # SESSION_TYPE, which the driver's `${SESSION_TYPE:-gpu_8x_a100_80gb_sxm4}` default
  # resolved to the 8x type. A rehearsal would have been launched on an 8x box by a script
  # written to avoid exactly that.
  PICK=$("$PICK_PY" - "$TYPES" "$MAX_RATE" "$PICKTMP" <<'PYPICK'
import json, sys
types, ceiling, path = sys.argv[1].split(), float(sys.argv[2]), sys.argv[3]
try:
    data = json.load(open(path)).get("data", {}) or {}
except Exception as exc:
    print("PICK_ERROR", exc); raise SystemExit(0)
for t in types:
    spec = data.get(t) or {}
    it = spec.get("instance_type") or {}
    regions = [r.get("name") for r in (spec.get("regions_with_capacity_available") or [])]
    price = it.get("price_cents_per_hour")
    if regions and price is not None and float(price) / 100.0 <= ceiling:
        print(t, "%.2f" % (float(price) / 100.0), regions[0])
        break
PYPICK
)
  case "$PICK" in PICK_ERROR*) say "WARN: $PICK";; esac
  CHOSEN=$(printf '%s' "$PICK" | awk 'NR==1{print $1}')
  RATE=$(printf '%s'   "$PICK" | awk 'NR==1{print $2}')
  REGION=$(printf '%s' "$PICK" | awk 'NR==1{print $3}')
  # HARD VALIDATION. An empty or unexpected type must never reach the driver, because the
  # driver's `${SESSION_TYPE:-...}` default is the 8x type.
  case " $TYPES " in
    *" $CHOSEN "*) [ -n "$RATE" ] && break ;;
  esac
  CHOSEN=""
  if [ "$(date -u +%s)" -ge "$END" ]; then
    say "NO_CAPACITY after $((DEADLINE_S/60)) min of polling; nothing launched, nothing billing"
    echo NO_CAPACITY > "$REPO/.pick_1x_no_capacity"; exit 0
  fi
  say "no 1x card from the list yet; re-polling (deadline in $(( (END - $(date -u +%s))/60 )) min)"
  sleep 150
done
rm -f "$PICKTMP"
say "CHOSEN $CHOSEN in $REGION at \$$RATE/hr (a rehearsal on this card: plumbing, not timing)"

# The driver still owns the launch, the timer, the terminate and the pull; it is told
# exactly one type, and a cutoff a few minutes out because capacity is available NOW.
export SESSION_TYPE="$CHOSEN"   # never empty: validated against the list above
export SESSION_RATE="$RATE"
export SESSION_CAP="${SESSION_CAP:-3}"
# THE TIMER SCALES WITH THE RATE, so "$3 for 60 minutes" stays true on a card that does
# not cost $1.99/hr: h100_pcie is $3.29/hr and a 60-minute timer there would be $3.29.
# 60 minutes on everything at or below $3.00/hr; otherwise the longest whole minute that
# fits the cap.
TM=${TIMER_MIN:-60}
FIT=$(python3 -c "print(int(float('$SESSION_CAP')/float('$RATE')*60))")
if [ "$FIT" -lt "$TM" ]; then TM=$FIT; fi
[ "$TM" -ge 5 ] || { say "FATAL: \$$RATE/hr leaves only ${TM} min inside \$$SESSION_CAP"; exit 2; }
export TIMER_MIN="$TM"
say "timer set to ${TIMER_MIN} min at \$$RATE/hr = \$$(python3 -c "print('%.2f' % (float('$TM')/60*float('$RATE')))") (cap \$$SESSION_CAP)"
export POLL_CUTOFF_UTC="$(date -u -v+15M +%FT%TZ 2>/dev/null || date -u -d '+15 minutes' +%FT%TZ)"
say "handing to scripts/mlsys_session_a.sh with SESSION_TYPE=$CHOSEN SESSION_RATE=$RATE POLL_CUTOFF_UTC=$POLL_CUTOFF_UTC"
exec bash scripts/mlsys_session_a.sh

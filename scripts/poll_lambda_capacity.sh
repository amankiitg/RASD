#!/usr/bin/env bash
# Lambda capacity poller — NOTIFY-ONLY, never auto-launches.
#
# Polls Lambda's instance-types API at a randomized 15-30 sec interval
# and emits a structured line on stdout when capacity for one of the
# watched SKUs becomes available. Designed to be wrapped by Monitor —
# each line is one notification.
#
# Output schema (one of):
#   CAPACITY:  sku=<sku> region=<region> price=$<price>/hr
#   NO_CAPACITY:  iteration=<N> elapsed=<seconds>      (heartbeat, every ~30 polls)
#   ERROR:  <description>
#   RESTART:  iteration=<N> elapsed=<seconds>           (self-restart every RESTART_AFTER_HOURS)
#
# Self-restart (default 3 hours) protects against any long-running
# state drift in curl, DNS resolver, or shell. Just exec's itself with
# the same env.
#
# Cache busting: adds ?_=<random> query param + Cache-Control: no-cache
# header on every poll. The earlier version of this script had no
# cache busting AND a broken `SKUS_ENV="$SKUS" echo ... | python3` line
# (the env var was set on `echo`, not on the piped python — silently
# returned empty results forever).
#
# Usage:
#   LAMBDA_API_KEY=<key> bash scripts/poll_lambda_capacity.sh
#
# Configurable env vars:
#   LAMBDA_API_KEY        required
#   SKUS                  space-separated, default "gpu_8x_a100_80gb_sxm4 gpu_8x_a100"
#   INTERVAL_MIN          min poll interval seconds (default 15)
#   INTERVAL_MAX          max poll interval seconds (default 30)
#   RESTART_AFTER_HOURS   self-exec this often (default 3)
#   HEARTBEAT_EVERY       emit NO_CAPACITY heartbeat every N polls (default 30)

set -uo pipefail

: "${LAMBDA_API_KEY:?LAMBDA_API_KEY env var required}"

SKUS=${SKUS:-"gpu_8x_a100_80gb_sxm4 gpu_8x_a100"}
INTERVAL_MIN=${INTERVAL_MIN:-15}
INTERVAL_MAX=${INTERVAL_MAX:-30}
RESTART_AFTER_HOURS=${RESTART_AFTER_HOURS:-3}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-30}

start_ts=$(date +%s)
restart_deadline=$((start_ts + RESTART_AFTER_HOURS * 3600))
iteration=0
prev_state="unknown"

while true; do
    iteration=$((iteration + 1))
    now=$(date +%s)
    elapsed=$((now - start_ts))

    # Self-restart to clear any accumulated state
    if [ "$now" -ge "$restart_deadline" ]; then
        echo "RESTART:  iteration=$iteration elapsed=${elapsed}s"
        # Pass the same env through to the new process
        exec env \
            LAMBDA_API_KEY="$LAMBDA_API_KEY" \
            SKUS="$SKUS" \
            INTERVAL_MIN="$INTERVAL_MIN" \
            INTERVAL_MAX="$INTERVAL_MAX" \
            RESTART_AFTER_HOURS="$RESTART_AFTER_HOURS" \
            HEARTBEAT_EVERY="$HEARTBEAT_EVERY" \
            bash "$0"
    fi

    # Cache-busting query param + header. Use $RANDOM (bash 1-32k) plus
    # the iteration count so successive calls have unique URLs even if
    # RANDOM repeats.
    cache_bust="$RANDOM-$iteration-$now"
    json=$(curl -sS \
        --max-time 12 \
        -H "Cache-Control: no-cache" \
        -H "Pragma: no-cache" \
        -u "$LAMBDA_API_KEY:" \
        "https://cloud.lambda.ai/api/v1/instance-types?_=$cache_bust" \
        2>/dev/null)

    if [ -z "$json" ]; then
        echo "ERROR:  curl returned empty (network issue?) iteration=$iteration"
        # short jittered sleep before retry
        sleep $(( INTERVAL_MIN + RANDOM % (INTERVAL_MAX - INTERVAL_MIN + 1) ))
        continue
    fi

    # Parse: pass SKUS via env (the bug in the prior version was placing
    # the env-var assignment ON the `echo` instead of on `python3`, so
    # the python subprocess saw an empty list and never matched anything).
    result=$(SKUS_ENV="$SKUS" python3 -c '
import json, os, sys
data = sys.stdin.read()
try:
    d = json.loads(data)
except Exception as e:
    print(f"PARSE_ERROR:{e}", end="")
    sys.exit(0)
skus = os.environ.get("SKUS_ENV", "").split()
hits = []
for sku in skus:
    spec = d.get("data", {}).get(sku, {})
    regions = [r["name"] for r in spec.get("regions_with_capacity_available", [])]
    if regions:
        price = spec.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
        hits.append(f"sku={sku} region={regions[0]} price=${price:.2f}/hr")
print("|".join(hits), end="")
' <<< "$json")

    if [[ "$result" == PARSE_ERROR:* ]]; then
        echo "ERROR:  ${result#PARSE_ERROR:} iteration=$iteration"
    elif [ -n "$result" ]; then
        # Capacity available — emit one line per hit (Monitor sees one event per line)
        IFS='|' read -ra hits_arr <<< "$result"
        for h in "${hits_arr[@]}"; do
            echo "CAPACITY:  $h iteration=$iteration"
        done
        prev_state="capacity"
    else
        # Heartbeat: only emit every Nth iteration to avoid spamming Monitor
        if [ "$prev_state" = "capacity" ]; then
            echo "NO_CAPACITY:  iteration=$iteration elapsed=${elapsed}s (transition: capacity gone)"
        elif [ $((iteration % HEARTBEAT_EVERY)) -eq 0 ]; then
            echo "NO_CAPACITY:  iteration=$iteration elapsed=${elapsed}s"
        fi
        prev_state="no_capacity"
    fi

    # Random 15-30 sec sleep
    sleep $(( INTERVAL_MIN + RANDOM % (INTERVAL_MAX - INTERVAL_MIN + 1) ))
done

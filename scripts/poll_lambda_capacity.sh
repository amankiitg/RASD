#!/usr/bin/env bash
# Lambda capacity poller — NOTIFY-ONLY, never auto-launches.
#
# Per project safety rule: instances >$15/hr require explicit human
# authorization before launching. This script polls Lambda's capacity
# API every INTERVAL seconds and exits 0 when capacity for any of the
# watched SKUs becomes available. The operator (or agent on operator's
# behalf) is then expected to ask for explicit go-ahead before issuing
# the launch.
#
# Usage:
#   LAMBDA_API_KEY=<key> bash scripts/poll_lambda_capacity.sh
#
# Tail in another terminal to watch live:
#   tail -f /tmp/lambda_poll.log
#
# Configurable env vars:
#   LAMBDA_API_KEY   required — set before invoking
#   SKUS             space-separated list (default: gpu_8x_a100 gpu_8x_a100_80gb_sxm4)
#   LOG              log path (default: /tmp/lambda_poll.log)
#   INTERVAL         poll interval seconds (default: 60)
#   MAX_WAIT_HOURS   give up after this many hours (default: 24)
#
# Exit codes:
#   0    capacity available — see $LOG for which SKU + region
#   1    LAMBDA_API_KEY not set
#   124  timed out

set -uo pipefail

: "${LAMBDA_API_KEY:?LAMBDA_API_KEY env var required}"

LOG=${LOG:-/tmp/lambda_poll.log}
INTERVAL=${INTERVAL:-60}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-24}
SKUS=${SKUS:-"gpu_8x_a100 gpu_8x_a100_80gb_sxm4"}

> "$LOG"
{
    echo "Lambda capacity poller — NOTIFY-ONLY, not launching"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Watching SKUs: $SKUS"
    echo "Interval: ${INTERVAL}s"
    echo "Max wait: ${MAX_WAIT_HOURS}h"
    echo ""
} | tee -a "$LOG"

deadline=$(($(date +%s) + MAX_WAIT_HOURS * 3600))

while [ "$(date +%s)" -lt $deadline ]; do
    json=$(curl -sS -u "$LAMBDA_API_KEY:" \
        https://cloud.lambda.ai/api/v1/instance-types 2>/dev/null || echo '{}')

    # Parse capacity, gracefully handle missing or malformed responses
    result=$(SKUS_ENV="$SKUS" echo "$json" | python3 -c '
import json, os, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
skus = os.environ.get("SKUS_ENV", "").split()
hits = []
for sku in skus:
    spec = d.get("data", {}).get(sku, {})
    regions = [r["name"] for r in spec.get("regions_with_capacity_available", [])]
    if regions:
        price = spec.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
        hits.append(f"{sku}@{regions[0]} (${price:.2f}/hr)")
print("|".join(hits))
' 2>/dev/null || echo "")

    ts=$(date -u +%H:%M:%S)
    if [ -n "$result" ]; then
        {
            echo ""
            echo "============================================================"
            echo "[$ts]  *** CAPACITY AVAILABLE *** "
            echo "  $result"
            echo "============================================================"
            echo "NOTIFY-ONLY mode. NOT launching."
            echo "Operator must explicitly authorize the launch given the"
            echo "hourly cost (>\$15/hr per project safety rule)."
        } | tee -a "$LOG"
        exit 0
    fi
    echo "[$ts] no capacity for: $SKUS" >> "$LOG"
    sleep "$INTERVAL"
done

echo "Timed out after ${MAX_WAIT_HOURS}h with no capacity" | tee -a "$LOG"
exit 124

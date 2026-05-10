#!/usr/bin/env bash
# Auto-execute Phase C end-to-end on a freshly-launched Lambda pod.
#
# Trigger: when scripts/poll_lambda_capacity.sh exits 0 (capacity hit),
# run this script. It handles the full pod lifecycle:
#
#   1. Launch a gpu_8x_a100 instance (40 GB SXM4 tier, $15.92/hr)
#      preferring europe-central-1 (validated for R6.5); fall back to
#      any region returned by the capacity check
#   2. Poll instance status until "active" + IP populated
#   3. SSH-set up: rsync repo, install deps, install -e .
#   4. Run scripts/phase_c_pod_session.sh inside nohup so it survives
#      our local SSH disconnect
#   5. Periodically check for completion (marker files + log tail)
#   6. scp results back to local repo
#   7. Terminate the pod
#   8. Commit + push the pod-side artifacts
#
# Authorization: per user's 2026-05-06 PM instruction
# ("when poller exits please launch and complete the tasks on POD and
#  don't wait for any of my confirmation"), this script is the
# authorized auto-launch path. It honors the project safety rule by
# capping at one launch per invocation; never auto-relaunches if the
# instance dies mid-session.
#
# Required env vars (set inline before invoking):
#   LAMBDA_API_KEY    — Lambda Cloud REST API key (from runpod_creds.md)
#   WANDB_API_KEY     — wandb credential (forwarded to pod)
#   HF_TOKEN          — HF gated-model token (forwarded to pod)
#   LOCAL_SSH_KEY     — path to private key matching `rasd-amank` Lambda key
#                       (default: ~/.ssh/id_ed25519)
#
# Usage:
#   bash scripts/auto_execute_phase_c.sh
#
# Logs to: results/phase_c/auto_execute.log

set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/phase_c
LOG=results/phase_c/auto_execute.log
exec > >(tee -a "$LOG") 2>&1

echo ""
echo "============================================================"
echo "Auto-execute Phase C — start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

: "${LAMBDA_API_KEY:?LAMBDA_API_KEY required (see runpod_creds.md)}"
: "${WANDB_API_KEY:?WANDB_API_KEY required}"
: "${HF_TOKEN:?HF_TOKEN required}"
LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519}"
[ -f "$LOCAL_SSH_KEY" ] || { echo "SSH key not found at $LOCAL_SSH_KEY"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Pick SKU + region with current capacity
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Find capacity ---"
SKU=""
REGION=""
caps_json=$(curl -sS -u "$LAMBDA_API_KEY:" \
    https://cloud.lambda.ai/api/v1/instance-types)
for sku in gpu_8x_a100 gpu_8x_a100_80gb_sxm4; do
    region=$(echo "$caps_json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
spec = d.get('data', {}).get('$sku', {})
regions = [r['name'] for r in spec.get('regions_with_capacity_available', [])]
# Prefer europe-central-1 (validated for R6.5); fall back to first available
for pref in ('europe-central-1', 'us-west-2', 'us-east-1'):
    if pref in regions:
        print(pref); sys.exit(0)
print(regions[0] if regions else '')
")
    if [ -n "$region" ]; then
        SKU="$sku"
        REGION="$region"
        break
    fi
done
if [ -z "$SKU" ]; then
    echo "FAIL: No 8x A100 capacity found at launch time. Capacity flickered."
    echo "      Re-run scripts/poll_lambda_capacity.sh and try again."
    exit 1
fi
echo "Selected: SKU=$SKU REGION=$REGION"

# ---------------------------------------------------------------------------
# 2. Launch the instance
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Launch instance ($SKU in $REGION) ---"
launch_resp=$(curl -sS -u "$LAMBDA_API_KEY:" \
    -X POST https://cloud.lambda.ai/api/v1/instance-operations/launch \
    -H "Content-Type: application/json" \
    -d "{
        \"region_name\": \"$REGION\",
        \"instance_type_name\": \"$SKU\",
        \"ssh_key_names\": [\"rasd-amank\"],
        \"name\": \"rasd-m4-phase-c-$(date +%s)\",
        \"quantity\": 1
    }")
echo "Launch response: $launch_resp"
INST_ID=$(echo "$launch_resp" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    ids = d.get('data', {}).get('instance_ids', [])
    print(ids[0] if ids else '')
except Exception:
    print('')
")
if [ -z "$INST_ID" ]; then
    echo "FAIL: Launch did not return an instance_id (capacity may have flickered)"
    exit 1
fi
echo "Launched: INST_ID=$INST_ID"

# Trap to ensure we always terminate, even on failure mid-session
cleanup_terminate() {
    if [ -n "${INST_ID:-}" ]; then
        echo ""
        echo "--- TRAP: terminating instance $INST_ID ---"
        curl -sS -u "$LAMBDA_API_KEY:" \
            -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
            -H "Content-Type: application/json" \
            -d "{\"instance_ids\": [\"$INST_ID\"]}" | tee -a "$LOG"
    fi
}
# Note: we do NOT trap on EXIT unconditionally, because we only want to
# terminate after a deliberate completion or failure. The explicit
# terminate at end-of-script handles success; uncaught failures fall
# through to manual cleanup (with the instance ID in the log).

# ---------------------------------------------------------------------------
# 3. Wait for active + IP
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Wait for instance to become active ---"
IP=""
for attempt in $(seq 1 60); do  # up to 30 min
    inst_json=$(curl -sS -u "$LAMBDA_API_KEY:" \
        "https://cloud.lambda.ai/api/v1/instances/$INST_ID")
    parsed=$(echo "$inst_json" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    inst = d.get('data', {})
    print(inst.get('status', '?'), inst.get('ip', ''))
except Exception:
    print('?', '')
")
    status=$(echo "$parsed" | awk '{print $1}')
    candidate_ip=$(echo "$parsed" | awk '{print $2}')
    echo "[attempt $attempt] status=$status ip=$candidate_ip"
    if [ "$status" = "active" ] && [ -n "$candidate_ip" ]; then
        IP=$candidate_ip
        break
    fi
    sleep 30
done
if [ -z "$IP" ]; then
    echo "FAIL: instance never reached 'active' status"
    cleanup_terminate
    exit 1
fi
echo "Active: IP=$IP"

# ---------------------------------------------------------------------------
# 4. SSH setup + run bundled session in nohup
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: SSH setup + start session ---"
SSH="ssh -i $LOCAL_SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# Wait for SSH to come up (sshd takes a beat after instance becomes active)
for attempt in $(seq 1 20); do
    if $SSH ubuntu@$IP -o ConnectTimeout=5 echo ok 2>/dev/null; then
        echo "SSH up after $attempt attempts"
        break
    fi
    echo "[ssh-wait $attempt] not ready yet"
    sleep 15
done

# Sanity check + inline env-var setup, then clone+install+launch
$SSH ubuntu@$IP "bash -s" <<EOF
set -euo pipefail
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
git clone https://github.com/amankiitg/RASD.git || true
cd RASD
git fetch --tags
git checkout main
git pull

# Conda env (Lambda has miniconda3 preinstalled at /home/ubuntu/miniconda3)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh && bash /tmp/mc.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q rasd-gpu; then
    # Accept Anaconda TOS — required since 2024 policy change for the
    # default pkgs/main + pkgs/r channels. (Hit on 2026-05-10 first
    # auto-execute attempt.)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
    # Bare conda env: just python, then pip the rest from lock file.
    # Replaces the M3 environment_gpu.yml flow because (a) conda's pip
    # subcall doesn't pass --no-build-isolation (flash-attn fails),
    # (b) conda's pytorch=2.1.0 gets re-upgraded by pip transitive
    # deps to torch 2.11+cu130 which doesn't run on driver 12.8,
    # (c) bitsandbytes 0.49.2 needs torch>=2.4. (All discovered on
    # 2026-05-10 attempts 2-3; cost ~\$10 of pod-time.)
    conda create -n rasd-gpu python=3.10 -y
fi
conda activate rasd-gpu
# Use the captured pod lock — pinned versions known to work end-to-end
# on this exact CUDA-driver / Lambda image combination.
pip install -r requirements-lock.txt
# flash-attn is in the lock but its wheel-build needs --no-build-isolation
# at first install. If the wheel was already built by `pip install -r`
# above, this is a no-op; if not, --no-build-isolation lets it find torch.
pip install --no-build-isolation "flash-attn==$(grep '^flash-attn' requirements-lock.txt | cut -d= -f3)" || true
pip install -e .

# Sanity: full test suite
pytest tests/ -q

# Launch the bundled session under nohup so it survives our SSH disconnect
mkdir -p results/phase_c
export WANDB_API_KEY="$WANDB_API_KEY"
export HF_TOKEN="$HF_TOKEN"
export HF_HOME=/home/ubuntu/hf_cache
nohup bash scripts/phase_c_pod_session.sh > results/phase_c/session.log 2>&1 &
echo "Session PID: \$!"
echo "\$!" > /tmp/phase_c_pid
EOF

if [ $? -ne 0 ]; then
    # Per 2026-05-10 user instruction: don't auto-terminate on
    # bootstrap failures. Keep the pod alive so the operator can
    # SSH in, fix the issue (e.g., conda yaml, pip install snag),
    # and re-run scripts/phase_c_pod_session.sh manually. Avoids
    # re-burning ~14 min of bootstrap on every iteration.
    echo "FAIL: SSH setup or bootstrap failed"
    echo ""
    echo "POD KEPT ALIVE for manual debug. To fix:"
    echo "  ssh -i $LOCAL_SSH_KEY ubuntu@$IP"
    echo "  # ... apply fix on the pod, then re-run:"
    echo "  cd RASD && nohup bash scripts/phase_c_pod_session.sh \\"
    echo "    > results/phase_c/session.log 2>&1 &"
    echo ""
    echo "When done (success or abandon), terminate manually:"
    echo "  curl -sS -u \"\$LAMBDA_API_KEY:\" \\"
    echo "    -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"instance_ids\": [\"$INST_ID\"]}'"
    echo ""
    echo "Instance: $INST_ID  IP: $IP"
    exit 1
fi
echo "Session running on pod under nohup"

# ---------------------------------------------------------------------------
# 5. Poll for completion (up to 14 hours)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: Poll for session completion ---"
deadline=$(($(date +%s) + 14 * 3600))
while [ "$(date +%s)" -lt $deadline ]; do
    # Check if the master script's last marker exists
    last_marker=$($SSH ubuntu@$IP "ls -1t RASD/results/phase_c/*.done 2>/dev/null | head -1 | xargs -n1 basename" 2>/dev/null || echo "")
    pid_alive=$($SSH ubuntu@$IP "kill -0 \$(cat /tmp/phase_c_pid 2>/dev/null) 2>/dev/null && echo alive || echo gone" 2>/dev/null || echo "?")
    log_tail=$($SSH ubuntu@$IP "tail -3 RASD/results/phase_c/session.log 2>/dev/null" 2>/dev/null || echo "")
    ts=$(date -u +%H:%M:%S)
    echo "[$ts] pid=$pid_alive last_marker=$last_marker"
    echo "      log_tail: $log_tail" | head -3
    # Done condition: pid gone AND p36 marker present
    if [ "$pid_alive" = "gone" ] && echo "$last_marker" | grep -q "p36"; then
        echo "Session complete (p36 marker present)"
        break
    fi
    if [ "$pid_alive" = "gone" ] && [ -z "$last_marker" ]; then
        echo "FAIL: session died before any marker — check session.log"
        $SSH ubuntu@$IP "tail -30 RASD/results/phase_c/session.log" || true
        cleanup_terminate
        exit 1
    fi
    sleep 600  # poll every 10 min
done

# ---------------------------------------------------------------------------
# 6. scp results back to local repo
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: scp results back ---"
SCP="scp -i $LOCAL_SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r"
for d in c11_validation yarn_validation c6_validation m4_smoke baselines final phase_c; do
    $SCP "ubuntu@$IP:~/RASD/results/$d" results/ 2>&1 | tail -5 || true
done
$SCP "ubuntu@$IP:~/RASD/requirements-lock.txt" . 2>&1 | tail -2 || true
$SCP "ubuntu@$IP:~/RASD/configs/ablations.yml" configs/ 2>&1 | tail -2 || true

# ---------------------------------------------------------------------------
# 7. Commit + push pod-side artifacts
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 7: Commit pod-side artifacts ---"
git add results/ requirements-lock.txt configs/ablations.yml 2>/dev/null || true
if git diff --cached --quiet; then
    echo "No new artifacts to commit (likely a re-run)"
else
    git commit -m "M4 Phase C — bundled pod session results

Auto-committed by scripts/auto_execute_phase_c.sh after the pod
session completed end-to-end. Stage markers under results/phase_c/
indicate which gates passed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    git tag -a m4-phase-c-complete -m "M4 Phase C — final matrix landed" || true
    git push origin main || true
    git push origin m4-phase-c-complete || true
fi

# ---------------------------------------------------------------------------
# 8. Terminate the pod
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 8: Terminate instance ---"
cleanup_terminate
echo ""
echo "============================================================"
echo "Auto-execute Phase C — done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

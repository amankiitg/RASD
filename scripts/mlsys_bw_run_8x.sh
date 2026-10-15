#!/usr/bin/env bash
# THE TORCH_NCCL_BLOCKING_WAIT DIAGNOSTIC — one 8x A100 80GB SXM4 session, ceiling $30.
#
# Usage:  bash scripts/mlsys_bw_run_8x.sh                     # the measured session
#         MLSYS_BW_REHEARSE=1 bash scripts/mlsys_bw_run_8x.sh # 1x plumbing rehearsal
#
# WHY A COMMITTED LAUNCHER. The session's parameters ARE its experiment: the same box, the
# same book, the same seed and the same engine configurations, differing only in the wait
# mode. Writing them here means the numbers in the report have a command line behind them.
#
# WHY THE WAIT MODE IS FORWARDED. Until today the pod's `.pod_env.sh` hard-coded
# `TORCH_NCCL_BLOCKING_WAIT=1`, and the remote scripts source that file with `set -a`, so the
# file always beat anything a driver sent. The file now derives its NCCL lines from
# MLSYS_NCCL_WAIT_MODE and records the mode as a comment only, so a forwarded value survives
# sourcing. The per-ARM override is applied inside the remote script with `env` / `env -u`,
# because ProcessGroupNCCL reads the flag once per process group and it therefore cannot
# change within one launch.
#
# WHAT COSTS WHAT. Two bare-ring benches (~5 min each: 256 rounds x 5 payloads x 2 timings),
# two engine-arm groups at 128k (3 arms x 2 modes, 16 and 32 tokens per row), and two 1M
# target-only rows (~5 min each under the async arm, longer under blocking). Roughly 30 min of
# GPU work behind ~10 min of boot and provisioning, so 100 min is work plus a wide margin.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REH=${MLSYS_BW_REHEARSE:-0}

if [ "$REH" = "1" ]; then
  # PLUMBING ONLY, one card, 8k, 8 rounds. `install_ring_attention` is a no-op at
  # world_size <= 1, so the engine arms cannot be measured here and the ring bench says so
  # itself rather than printing a timing for a no-op. What this run DOES prove, and why it is
  # worth $1-2 before spending 8x money:
  #   * the arm environment reaches the process (`env` sets, `env -u` clears) and every row
  #     records `env_arm` plus an `nccl_env` whose wait_mode matches the arm;
  #   * the two arms are two launches and both write rows;
  #   * the parameter contract accepts this driver's manifest, the puller works, and the DONE
  #     marker carries the intended rc.
  SESSION_TYPE=gpu_1x_a100_sxm4
  SESSION_OUT=results/mlsys/bw_reh
  SESSION_TAG=bw_reh
  SESSION_CFG=configs/mlsys_bw_diag.yml
  SESSION_SANITY_CFG=configs/mlsys_bw_diag.yml
  SESSION_RATE=1.29
  SESSION_CAP=3
  NPROC=1
  TIMER_MIN=60
  BW_ROUNDS=8
  BW_CTX=8192
  BW_1M_OK=0
  BW_ARMS=ring2048
  BW_TARGET_TOKENS=8
  BW_SPEC_TOKENS=16
  BW_SINGLE_RANK_OK=1
  # Both arms, because the plumbing under test IS the arm switch.
  BW_ENV_ARMS="blocking async"
else
  SESSION_TYPE=gpu_8x_a100_80gb_sxm4
  SESSION_OUT=results/mlsys/bw_diag
  SESSION_TAG=bw_diag
  SESSION_CFG=configs/mlsys_bw_diag.yml
  SESSION_SANITY_CFG=configs/mlsys_bw_diag.yml
  SESSION_RATE=22.32
  SESSION_CAP=30
  NPROC=8
  TIMER_MIN=100                # work + 30, per the standing rule
  BW_ROUNDS=256
  BW_CTX=131072
  BW_1M_OK=1
  BW_ARMS=ring2048,ring32768,splitkv
  BW_TARGET_TOKENS=16
  BW_SPEC_TOKENS=32
  BW_SINGLE_RANK_OK=0
  BW_ENV_ARMS="blocking async"
fi

export SESSION_TYPE SESSION_OUT SESSION_TAG SESSION_CFG SESSION_SANITY_CFG
export SESSION_RATE SESSION_CAP NPROC TIMER_MIN
export REMOTE_SCRIPT=scripts/mlsys_bw_remote.sh
export MLSYS_PULL_EVERY_S=240

export BW_OUT="$SESSION_OUT"
export BW_BOOK=pg19_train_0
export BW_SEED=42
export BW_CTX_1M=1048576
export BW_ENV_ARMS
export BW_SINGLE_RANK_OK

# The mode the pod's env file is provisioned with. The per-arm override inside the remote
# script is what the experiment actually uses; this is the session-level default, forwarded so
# the pod can log it and so a reader can see the intended mode.
export MLSYS_NCCL_WAIT_MODE=async

# FORWARD_ENV is fail-fast: a name that is unset stops the launch before an instance exists.
export FORWARD_ENV="BW_OUT BW_BOOK BW_SEED BW_ROUNDS BW_CTX BW_CTX_1M BW_ARMS BW_TARGET_TOKENS BW_SPEC_TOKENS BW_1M_OK BW_ENV_ARMS BW_SINGLE_RANK_OK MLSYS_NCCL_WAIT_MODE"

case "$REH" in
  1) EXPECT_BW_1M_OK=0; EXPECT_ARMS=ring2048; EXPECT_TGT=8; EXPECT_SPEC=16 ;;
  *) EXPECT_BW_1M_OK=1; EXPECT_ARMS=ring2048,ring32768,splitkv; EXPECT_TGT=16; EXPECT_SPEC=32 ;;
esac
export REMOTE_EXPECT_EXTRA="BW_OUT=$SESSION_OUT;BW_BOOK=pg19_train_0;BW_SEED=42;BW_ROUNDS=$BW_ROUNDS;BW_CTX=$BW_CTX;BW_CTX_1M=1048576;BW_ARMS=$EXPECT_ARMS;BW_TARGET_TOKENS=$EXPECT_TGT;BW_SPEC_TOKENS=$EXPECT_SPEC;BW_1M_OK=$EXPECT_BW_1M_OK;BW_ENV_ARMS=$BW_ENV_ARMS;BW_SINGLE_RANK_OK=$BW_SINGLE_RANK_OK"

echo "bw mode=$([ "$REH" = 1 ] && echo rehearsal || echo measured) type=$SESSION_TYPE out=$SESSION_OUT nproc=$NPROC timer=${TIMER_MIN}min cap=\$$SESSION_CAP arms='$BW_ENV_ARMS'"
bash scripts/mlsys_session_a.sh

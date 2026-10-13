#!/usr/bin/env bash
# THE PART 1 DIAGNOSTIC SESSION — 8x A100 80GB SXM4, one instance, ceiling $35.
#
# Usage:  bash scripts/mlsys_df_run_8x.sh
#
# WHY A COMMITTED LAUNCHER. The session's parameters ARE its experiment: four engine
# configurations compared against each other, and a bare-ring measurement that only means
# something if the payloads are the ones the engine actually ships. Writing them here means
# the run is reproducible from the repo, the numbers in the report have a command line
# behind them, and a later reader can tell which knobs were set without reading a shell
# history.
#
# THE STAGE ORDER IS THE PRIORITY ORDER (see scripts/mlsys_df_remote.sh): topo, the bare
# ring bench, the split-KV smoke, the engine arms, the ppl check. Only the last two depend on
# each other, through marker files, so a failure anywhere still leaves the earlier evidence.
#
# DEAD-MAN TIMER. TIMER_MIN caps the session's wall clock and is the only thing that
# terminates the pod if the driver dies; the ceiling is enforced separately by the driver
# against the instance's own creation time. work here is ~25 min of GPU time on top of
# ~10 min of boot and provisioning, so 75 min is work + a wide margin and still $27.9 of the
# $35 ceiling.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SESSION_TYPE=gpu_8x_a100_80gb_sxm4 \
SESSION_OUT=results/mlsys/decode_df \
SESSION_TAG=decode_df \
SESSION_CFG=configs/mlsys_df_followup.yml \
SESSION_SANITY_CFG=configs/mlsys_df_followup.yml \
SESSION_RATE=22.32 \
SESSION_CAP=35 \
REMOTE_SCRIPT=scripts/mlsys_df_remote.sh \
NPROC=8 \
TIMER_MIN=75 \
MLSYS_PULL_EVERY_S=240 \
DF_COMMIT="$(git rev-parse HEAD)" \
FORWARD_ENV="DF_COMMIT DF_OUT DF_CTX_ARMS DF_CTX_SMOKE DF_BOOK DF_ARMS DF_SEED DF_BENCH_ROUNDS DF_TARGET_TOKENS DF_SPEC_TOKENS DF_SINGLE_RANK_OK" \
DF_OUT=results/mlsys/decode_df \
DF_CTX_ARMS=131072 \
DF_CTX_SMOKE=8192 \
DF_BOOK=pg19_train_0 \
DF_ARMS=ring2048,ring32768,ring32768x,splitkv \
DF_SEED=42 \
DF_BENCH_ROUNDS=256 \
DF_TARGET_TOKENS=16 \
DF_SPEC_TOKENS=32 \
DF_SINGLE_RANK_OK=0 \
REMOTE_EXPECT_EXTRA="DF_COMMIT=$DF_COMMIT;DF_OUT=results/mlsys/decode_df;DF_CTX_ARMS=131072;DF_CTX_SMOKE=8192;DF_BOOK=pg19_train_0;DF_ARMS=ring2048,ring32768,ring32768x,splitkv;DF_SEED=42;DF_BENCH_ROUNDS=256;DF_TARGET_TOKENS=16;DF_SPEC_TOKENS=32;DF_SINGLE_RANK_OK=0" \
bash scripts/mlsys_session_a.sh

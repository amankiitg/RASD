#!/usr/bin/env bash
# PART 3 — the headline rerun under the candidate engine (split-KV).
#
# Usage:  bash scripts/mlsys_rerun_run_8x.sh {r1|r2|rehearse}
#
# WHY THE STANDARD REMOTE SCRIPT RATHER THAN THE DF ONE. The brief says the rerun keeps Session
# A's protocol in every field except the candidate's changes, and `mlsys_session_a_remote.sh` IS
# that protocol's implementation: the sanity pair that must produce 2 ok rows before anything
# else, the per-stage chain, the degenerate-row inspection driven by SESSION_CHECK_STAGE, the
# incremental pulls. Reimplementing it for this run would change the protocol, which is the one
# thing that must not change. So the candidate lives entirely in the CONFIG (its `defaults`
# carry ring_decode_mode / kv_block_size / ring_expand_after_rotation), and the driver is the
# same driver every previous headline session used.
#
# R1 AND R2 ARE SEPARATE LAUNCHES ON PURPOSE. R1 carries the gate: its first two rows are a
# 128k spec+target pair whose TTFT must match the old engine (S3 stops if it is off by more
# than 10%), and whose measured rates project the rest of the campaign. Starting R2 before that
# gate is read would spend the ceiling on an engine nobody had checked. R2 may then run in
# PARALLEL with R1 on a second 8x box if capacity allows, which is why they are two launches
# rather than two stages of one.
#
# DEAD-MAN TIMER (R7): ~105 min of work behind ~10 min of boot and provisioning, so 170 min =
# work x 1.5, and $63 of the $200 ceiling per session. Projected total for both: ~$79.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE=${1:-}
# PARALLEL LAUNCHES. The driver asserts that it owns exactly one instance and refuses to start
# if any other is present, unless the others are declared positively through
# EXPECTED_OTHER_INSTANCES. That guard is worth keeping -- it is what stops a launch against a
# stale box -- so R2 declares R1's instance rather than disabling the check.
#   EXPECTED_OTHER_INSTANCES=<id> bash scripts/mlsys_rerun_run_8x.sh r2
export EXPECTED_OTHER_INSTANCES=${EXPECTED_OTHER_INSTANCES:-}
case "$MODE" in
  r1)
    SESSION_TYPE=gpu_8x_a100_80gb_sxm4
    SESSION_OUT=results/mlsys/rerun_r1
    SESSION_TAG=rerun_r1
    SESSION_CFG=configs/mlsys_rerun_r1.yml
    SESSION_SANITY_CFG=configs/mlsys_rerun_sanity.yml
    SESSION_CHECK_STAGE=RR_GATE_128K_SPEC
    SESSION_RATE=22.32
    SESSION_CAP=70
    NPROC=8
    TIMER_MIN=170
    ;;
  r2)
    SESSION_TYPE=gpu_8x_a100_80gb_sxm4
    SESSION_OUT=results/mlsys/rerun_r2
    SESSION_TAG=rerun_r2
    SESSION_CFG=configs/mlsys_rerun_r2.yml
    SESSION_SANITY_CFG=configs/mlsys_rerun_sanity.yml
    SESSION_CHECK_STAGE=RR_SEEDS_1M_SPEC
    SESSION_RATE=22.32
    SESSION_CAP=70
    NPROC=8
    TIMER_MIN=170
    ;;
  rehearse)
    # PLUMBING ONLY, on one card, at 8k. This exists to prove that the new engine provenance
    # columns (ring_decode_mode / ring_chunk_size / ring_expand_after_rotation /
    # ring_layers_patched / git_commit / nccl_env) reach a CSV row through the STANDARD runner,
    # because stop condition S2 voids the rerun if a row cannot prove which engine produced it.
    SESSION_TYPE=gpu_1x_a100_sxm4
    SESSION_OUT=results/mlsys/rerun_reh
    SESSION_TAG=rerun_reh
    SESSION_CFG=configs/mlsys_rerun_rehearsal.yml
    SESSION_SANITY_CFG=configs/mlsys_rerun_rehearsal.yml
    SESSION_CHECK_STAGE=
    SESSION_RATE=1.29
    SESSION_CAP=3
    NPROC=1
    TIMER_MIN=60
    ;;
  *)
    echo "usage: $0 {r1|r2|rehearse}" >&2; exit 2;;
esac
export SESSION_TYPE SESSION_OUT SESSION_TAG SESSION_CFG SESSION_SANITY_CFG
export SESSION_CHECK_STAGE SESSION_RATE SESSION_CAP NPROC TIMER_MIN
export MLSYS_PULL_EVERY_S=${MLSYS_PULL_EVERY_S:-240}
echo "rerun mode=$MODE type=$SESSION_TYPE out=$SESSION_OUT cfg=$SESSION_CFG nproc=$NPROC timer=${TIMER_MIN}min cap=\$$SESSION_CAP"
bash scripts/mlsys_session_a.sh

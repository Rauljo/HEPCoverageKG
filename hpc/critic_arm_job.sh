#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=arm
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/arm_%j.out
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# =============================================================================
# One arm of the critic re-measurement (D-062 follow-up).
#
#   ARM=off       the control: the ordinary code path, no critic
#   ARM=ranked    critic on, candidates in retrieval order (the current default)
#   ARM=shuffled  critic on, candidates shuffled -- the default is in doubt
#
#   CONTRACT=v1   the answer as it has always been (default)
#   CONTRACT=v2   cite a set instead of retyping ids, `refine`, and an
#                 abstention challenged for its reasons
#
# The two cross: {critic off,on} x {answer v1,v2} is a 2x2, and the interaction
# is the interesting cell -- if citing works, the critic's filtering matters MORE,
# because a cited set becomes the answer directly instead of being a pool the
# model picks from.
#
# repeats=3, because D-062's +0.025 had no error bar and S-52 says three repeats
# before comparing anything.
#
# All three arms run CONCURRENTLY so they meet the same server. That makes
# `seconds` contaminated by contention and not comparable across arms in this
# run -- D-062 already has clean timings at a fixed 9-wide load, and correctness
# does not depend on queueing.
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
export LLM_BASE_URL="http://compute-gpu-0-1:${VLLM_PORT:-8000}/v1"
QUESTIONS="$1"; ARM="${ARM:-off}"
case "$ARM" in
  off)      FLAGS="" ;;
  ranked)   FLAGS="--critic" ;;
  shuffled) FLAGS="--critic --critic-seed 20260815" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
case "${CONTRACT:-v1}" in
  v1) ;;
  v2) FLAGS="$FLAGS --answer-contract" ;;
  *) echo "unknown CONTRACT=$CONTRACT"; exit 2 ;;
esac
echo "host=$(hostname)  arm=$ARM  contract=${CONTRACT:-v1}  questions=$QUESTIONS  flags='$FLAGS'"
.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system planner --repeats 3 $FLAGS

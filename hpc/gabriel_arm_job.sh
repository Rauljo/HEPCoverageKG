#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=garm
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/garm_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# =============================================================================
# One arm of the critic ablation, scored against the SUPERVISOR'S gold.
#
# FIVE ARMS, NOT EIGHT. The obvious design is 2x2x2 over {critic, ordering,
# judge}, but with the critic OFF there is no ordering and no judge -- those
# knobs do not exist on that path. Running them would produce three pairs of
# byte-identical results and cost 3/8 of the compute to learn nothing.
#
#   ARM=off             the control: no critic
#   ARM=ranked-72b      critic on, retrieval order,  judge = the answerer
#   ARM=ranked-8b       critic on, retrieval order,  judge = Llama-3.1-8B
#   ARM=shuffled-72b    critic on, shuffled,         judge = the answerer
#   ARM=shuffled-8b     critic on, shuffled,         judge = Llama-3.1-8B
#
# EVERYTHING ELSE IS HELD AT THE PROPOSED FREEZE: contract v3 and
# force-critic-set on. One axis at a time, or the result is uninterpretable --
# D-062 lost a week to a one-line prompt change that moved the control.
# `force-critic-set` is a no-op without a critic, so the off arm is unaffected
# and "critic off" vs "critic on" remains a clean comparison of the pipeline as
# it would ship.
#
# THE GOLD IS PARTIAL, BY CONSTRUCTION. Gabriel only ever saw papers our system
# surfaced, so `judged_set_f1` scores inside the judged universe -- a paper he
# never looked at is unknown, not wrong. Precision under that restriction is
# clean; recall is optimistic and cannot be otherwise. See D-072.
#
# EIGHT QUESTIONS IS THIN. The power analysis (D-063) wanted 200 for a 0.0008
# effect. Nothing here will resolve a small difference, and it is not meant to:
# it is the only measurement in the project against labels a physicist wrote.
# Run QUESTIONS=eval/questions/dev-2026-08-03-paperA-200.jsonl for the powered
# version, and read the two together.
#
# The judge endpoint is chosen by CRITIC_BASE_URL: unset means the critic talks
# to the planner's own client (the 72B), set points it at the small model. This
# is why serve_split.sh must be up for the -8b arms and why they will fail fast,
# not silently, if it is not.
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG

GPU_HOST="${GPU_HOST:-compute-gpu-0-1}"
export LLM_BASE_URL="http://${GPU_HOST}:${VLLM_PORT:-8000}/v1"

QUESTIONS="${1:-eval/questions/gabriel-gold-2026-08-25.jsonl}"
ARM="${ARM:-off}"
REPEATS="${REPEATS:-3}"

# Held fixed across every arm.
FLAGS="--contract v3"

case "$ARM" in
  off)
    ;;
  ranked-72b)
    FLAGS="$FLAGS --critic --force-critic-set" ;;
  shuffled-72b)
    FLAGS="$FLAGS --critic --force-critic-set --critic-seed 20260815" ;;
  ranked-8b)
    FLAGS="$FLAGS --critic --force-critic-set"
    export CRITIC_BASE_URL="http://${GPU_HOST}:8001/v1"
    export CRITIC_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
  shuffled-8b)
    FLAGS="$FLAGS --critic --force-critic-set --critic-seed 20260815"
    export CRITIC_BASE_URL="http://${GPU_HOST}:8001/v1"
    export CRITIC_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

# Fail before spending a GPU hour rather than after. A missing 8B endpoint on a
# -8b arm silently falls back to the 72B in `_critic_client`, which would put
# two different arms under one label -- the exact failure D-070 exists to stop.
case "$ARM" in
  *-8b)
    if ! curl -sf --max-time 10 "${CRITIC_BASE_URL}/models" >/dev/null; then
      echo "FATAL: judge endpoint ${CRITIC_BASE_URL} is not answering."
      echo "       serve_split.sh must be running, or this arm would quietly"
      echo "       run the 72B and be recorded as the 8B."
      exit 3
    fi ;;
esac

echo "host=$(hostname)  arm=$ARM  questions=$QUESTIONS  repeats=$REPEATS"
echo "flags='$FLAGS'  critic_endpoint='${CRITIC_BASE_URL:-<planner client, 72B>}'"

.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system planner --repeats "$REPEATS" $FLAGS

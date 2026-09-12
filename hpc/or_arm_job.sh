#!/usr/bin/env bash
#SBATCH --job-name=orarm
# compute-0-1 ACCEPTS WORK AND DOES NONE -- every other job script in this
# directory excludes it and this one did not. 2026-09-12: two OpenRouter arms
# landed there, showed RUNNING in squeue for eight minutes, and produced no
# log file, no process and no run file. The node takes the allocation and
# never starts the step.
#SBATCH --exclude=compute-0-1
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/orarm_%j.out
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# ONE ARM, ONE JOB, answers from OpenRouter. Parallel by default so a batch is
# minutes not hours, and so a stopped laptop session cannot take it down.
# D-093: the control is submitted TWICE in every batch, in parallel, because two
# identical QwQ runs differed by 0.113 while the whole arm spread was 0.048.
# A batch that does not price its own noise cannot be read.
set -uo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
_OVERRIDE_CRITIC="${CRITIC_MODEL:-}"
set -a; . ./.env; set +a
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_API_KEY="$OPENROUTER_API_KEY"
export CRITIC_BASE_URL="https://openrouter.ai/api/v1"
export CRITIC_API_KEY="$OPENROUTER_API_KEY"
export LLM_MODEL_NAME="${OR_MODEL:-qwen/qwen3-32b}"
# THE SUBMITTED JUDGE MUST WIN (D-082, third instance). This line used to be
# unconditional, so `--export=...,CRITIC_MODEL=qwen/qwen3.5-9b` was silently
# discarded and fourteen arms launched on 2026-09-12 to remove a judge
# confound all ran with the Llama-8B anyway. gabriel_arm_job.sh has carried an
# _OVERRIDE_CRITIC guard for this since 2026-09-01; this script never got one.
export CRITIC_MODEL="${_OVERRIDE_CRITIC:-meta-llama/llama-3.1-8b-instruct}"
unset LLM_MAX_COMPLETION_TOKENS
export KIND_SEMANTICS="${KIND_SEMANTICS:-0}"
[ -n "${LLM_API_KEY:-}" ] || { echo "FATAL: no OPENROUTER_API_KEY"; exit 2; }
Q="${QFILE:-eval/questions/gabriel-gold-2026-09-02-merged.jsonl}"
echo "ARM=${ARM_LABEL:-unnamed}  flags=${ARM_FLAGS:-baseline}  repeats=${REPS:-5}"
echo "questions=$Q  model=$LLM_MODEL_NAME  encoder=${ALIASES_EMBED_MODEL:-bge-base}"
echo "judge=$CRITIC_MODEL  constrained=${CONSTRAINED_IDS:-0} critic_selects=${CRITIC_SELECTS:-0} seq=${SUBGOAL_SEQUENTIAL:-0} chain=${SUBGOAL_CHAIN:-0}"
# --workers: the runner was serial until 2026-09-02 (D-094). A question is
# ~100s of network wait, so overlapping them is close to linear; the wall
# becomes the slowest question rather than the sum. Each worker builds its own
# system and sqlite connection -- required, because free-SQL keeps search sets
# in temp tables named on the connection.
.venv/bin/python -m hepcoveragekg.cli eval run "$Q" $ARM_FLAGS \
    --repeats "${REPS:-5}" --workers "${WORKERS:-8}" \
    --critic-seed 20260815 --timeout 1200 2>&1 | tail -26

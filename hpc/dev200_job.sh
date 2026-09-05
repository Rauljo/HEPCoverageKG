#!/usr/bin/env bash
#SBATCH --job-name=dev200
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/dev200_%j.out
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# ONE ARM, ONE JOB. Queued separately rather than chained so a single failure
# cannot take the batch down.
#
# WHY dev-200 AND NOT THE SUPERVISOR SET. gabriel-gold is EIGHT questions read
# three times: repeats shrink variance but add no question diversity, which is
# why the noise floor sits near 0.06 and swallows almost every effect measured
# on 2026-09-01. dev-2026-08-03-paperA-200 is 200 DISTINCT questions (93 set,
# 107 count) -- 25x the diversity. repeats=1 on purpose: 200 independent
# questions is worth far more than 8 questions seen three times, at the same
# cost.
set -uo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
set -a; . ./.env; set +a
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_API_KEY="$OPENROUTER_API_KEY"
export CRITIC_BASE_URL="https://openrouter.ai/api/v1"
export CRITIC_API_KEY="$OPENROUTER_API_KEY"
export LLM_MODEL_NAME="qwen/qwen3-32b"
export CRITIC_MODEL="meta-llama/llama-3.1-8b-instruct"
unset LLM_MAX_COMPLETION_TOKENS
export KIND_SEMANTICS="${KIND_SEMANTICS:-0}"
[ -n "${LLM_API_KEY:-}" ] || { echo "FATAL: no OPENROUTER_API_KEY"; exit 2; }
echo "ARM=${ARM_LABEL:-unnamed}  flags=${ARM_FLAGS:-none}"
echo "encoder=${ALIASES_EMBED_MODEL:-bge-base (default)}  kind_semantics=$KIND_SEMANTICS"
.venv/bin/python -m hepcoveragekg.cli eval run \
    eval/questions/dev-2026-08-03-paperA-200.jsonl \
    $ARM_FLAGS --repeats 1 --critic-seed 20260815 --timeout 1200 2>&1 | tail -30

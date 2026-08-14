#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=critic-eval
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/critic_eval_%j.out
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# =============================================================================
# The critic arm: the same three question sets, with --critic.
#
# The control arm is jobs 48426-8, run earlier today on the same code with the
# flag off -- which is the ordinary code path, not a second implementation, so
# the two differ by exactly one thing.
#
# Slower than the control by construction: every `search` now costs four extra
# model calls, and a search whose tail is still relevant widens and pays again.
# The time limit is 10h for that reason, not because anything is expected to
# hang.
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
export LLM_BASE_URL="http://compute-gpu-0-1:${VLLM_PORT:-8000}/v1"
QUESTIONS="$1"
echo "host=$(hostname)  questions=$QUESTIONS  CRITIC ON"
.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system planner --repeats 1 --critic --tag "critic-on"

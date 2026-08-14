#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=critic-arms
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/critic_arms_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# =============================================================================
# The two controls, then the four ordering arms (D-060).
#
# Runs AFTER the baseline, not beside it: the arms and the baseline share one
# vLLM endpoint, and the baseline records per-question timings that a second
# client would quietly inflate.
#
# The controls gate the arms. If the critic cannot say "keep all of these" and
# "drop all of these" on constructed cases, the flip rates below are noise about
# a model that is not judging, and the prompt is what needs work.
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
export LLM_BASE_URL="http://compute-gpu-0-1:${VLLM_PORT:-8000}/v1"
echo "host=$(hostname)"
.venv/bin/python -m hepcoveragekg.eval.critic_bias \
    --cases "${CASES:-40}" --out "eval/runs/critic-arms-${SLURM_JOB_ID}.txt"

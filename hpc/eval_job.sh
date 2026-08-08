#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=hepkg-eval
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/eval_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# =============================================================================
# Run the evaluation harness against the vLLM server on the GPU node.
#
# NOT on the login node: it is resource-capped and shared, and a multi-hour job
# there is both antisocial and likely to be killed.
#
# The venv is built against the Python 3.9 easybuild module, so the module MUST
# be loaded first -- otherwise `.venv/bin/python` dies with
# "libpython3.9.so.1.0: cannot open shared object file".
#
# LLM_BASE_URL points at the GPU node directly; compute nodes reach each other,
# so no SSH tunnel is involved and nothing depends on a laptop staying awake.
#
# compute-0-1 is EXCLUDED. On 2026-08-03 two eight-hour jobs landed there and
# produced no output file at all -- Slurm always creates one if the script runs,
# so nothing ran. Confirmed by hand: `srun -w compute-0-1 hostname` hangs
# indefinitely while the same command on compute-0-0 returns instantly. Same
# shape as the faulty A100 (D-040): a resource that accepts work and does none.
# Remove this line once the node is fixed or drained.
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG

# Port is an argument because two models are served at once: the rewriter on
# 8000 and the answerer on 8001. Hardcoding 8000 would evaluate the system using
# the wrong model and the numbers would look plausible.
export LLM_BASE_URL="http://compute-gpu-0-1:${VLLM_PORT:-8000}/v1"

QUESTIONS="$1"
REPEATS="${2:-1}"
TAG="${3:-shape}"

echo "host=$(hostname)  questions=$QUESTIONS  repeats=$REPEATS"
curl -s -m 10 -o /dev/null -w "endpoint: %{http_code}\n" "$LLM_BASE_URL/models" || true

.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system planner --repeats "$REPEATS" --tag "$TAG"

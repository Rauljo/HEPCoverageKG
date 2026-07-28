#!/bin/bash
#SBATCH --job-name=hepckg_pipe
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/pipeline_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# =============================================================================
# HEPCoverageKG: aliases Tiers 2/3 (deep semantics) on the DIAS SLURM cluster.
#
# Credentials are NEVER hardcoded here. LLM_BASE_URL / LLM_API_KEY /
# LLM_MODEL_NAME are read from a .env file in the repo root (gitignored), which
# hepcoveragekg/aliases/adjudicate.py also loads via python-dotenv. Copy your
# local .env to the cluster once; this script only validates that it is there.
# =============================================================================

set -euo pipefail

# Load modern Python module
module load Python/3.9.6-GCCcore-11.2.0

cd /home/xucabrjs/HEPCoverageKG

# Activate venv (pre-installed on login node -- compute nodes have no outbound network)
source .venv/bin/activate

# Load credentials from .env (never committed); -a exports them to the child process.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

: "${LLM_BASE_URL:?not set -- add it to .env on the cluster}"
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"
: "${LLM_MODEL_NAME:?not set -- add it to .env on the cluster}"

echo "Using LLM endpoint: ${LLM_BASE_URL} (model ${LLM_MODEL_NAME})"

# Run the python script
python3 scratch_test.py

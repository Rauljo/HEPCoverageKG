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

# The Tier-2 encoder must already be in the HF cache: compute nodes cannot reach
# the hub. Pre-fetch once on the LOGIN node with:
#   python3 -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-base-en-v1.5')"
export ALIASES_EMBED_MODEL="${ALIASES_EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
export HF_HUB_OFFLINE=1
echo "Using embedding model: ${ALIASES_EMBED_MODEL} (offline mode)"

# Tiers 2/2.5/3 only. Tiers 1+1.5 are deterministic and offline -- run those
# locally with `hepcoveragekg aliases build`, no cluster needed.
# --deep-out is explicit: never let the output path depend on the working directory.
# Concurrency 100 is safe here because the vLLM server is ours alone; the default
# is 8, which is what a shared rate-limited API can take.
python3 -m hepcoveragekg.cli aliases deep \
    --deep-out "data/processed/aliases_proposed_${SLURM_JOB_ID:-manual}.json" \
    --concurrency 100

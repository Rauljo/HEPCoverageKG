#!/bin/bash
#SBATCH -p LIGHTGPU
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=vllm-qwen72b
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm72b_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16

# =============================================================================
# Serve Qwen2.5-72B-Instruct for aliases Tier 3 adjudication.
#
# fp16, not quantised: ~145GB of weights across 4x A100-80GB via tensor
# parallelism, which leaves ample room for KV cache. The cluster has 6 idle
# A100s on LIGHTGPU, so there is no reason to accept AWQ's quality loss.
# 72B has 64 attention heads, so TP=2/4/8 all divide cleanly.
#
# Why a bigger model at all: the 8B was measured to answer "are these related?"
# rather than "are these the same?" -- it merged 8 distinct SMEFT Wilson
# coefficients, and accepted WH/ZH, W->e-nu / W->mu-nu and SRA/SRC as identical
# at confidence >= 0.9. See vault/logs/2026-07-28.md.
#
# The model must already be in the HF cache; compute nodes have no outbound
# network. Pre-fetch on the LOGIN node:
#   python -c "from huggingface_hub import snapshot_download; \
#              snapshot_download('Qwen/Qwen2.5-72B-Instruct')"
# =============================================================================

set -euo pipefail

module load Python/3.9.6-GCCcore-11.2.0

MODEL="${LLM_MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}"
PORT="${LLM_PORT:-8000}"

echo "Node:  $(hostname)"
echo "Model: ${MODEL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Credentials come from .env (gitignored), never hardcoded here.
cd /home/xucabrjs/HEPCoverageKG
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"

export HF_HUB_OFFLINE=1

apptainer exec --nv \
    ~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" \
    --api-key "${LLM_API_KEY}" \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

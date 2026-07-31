#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=vllm-qwen72b
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm72b_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16

# =============================================================================
# Serve Qwen2.5-72B-Instruct for aliases Tier 3 adjudication.
#
# AWQ 4-bit across 2x A100-80GB, on the GPU partition. The two GPU nodes are
# NOT interchangeable, which cost a failed job to discover:
#   LIGHTGPU / compute-gpu-0-0 : 6x MIG slices of 20GB. UNUSABLE for vLLM
#       0.8.5 at ANY size -- corrected 2026-07-31. An earlier note here guessed
#       "fine for the old 8B"; that was never observed (the 8B always ran on
#       -p GPU with a real A100) and jobs 48123/48124 disproved it. A single
#       slice fails in get_device_capability() -> nvmlDeviceGetHandleByIndex()
#       with NVMLError_InvalidArgument, BEFORE any weights load: Slurm puts a
#       MIG *UUID* in CUDA_VISIBLE_DEVICES and vLLM resolves it as a plain
#       device index. VLLM_USE_V1=0 does not help; the call is reached on
#       other paths too.
#   GPU / compute-gpu-0-1 : 3x real A100 80GB, no MIG. This is the one to use.
#
# Why AWQ rather than fp16: 72B fp16 is ~145GB of weights, and TP must divide
# the 64 attention heads, so TP=3 is invalid -- leaving TP=2 on 160GB, which
# fits the weights but leaves almost nothing for KV cache. AWQ is ~40GB, so TP=2
# leaves ~120GB of cache and much higher throughput, for a 1-2% quality cost
# that a binary same/different judgement will not notice.
#
# Why a bigger model at all: the 8B was measured to answer "are these related?"
# rather than "are these the same?" -- it merged 8 distinct SMEFT Wilson
# coefficients, and accepted WH/ZH, W->e-nu / W->mu-nu and SRA/SRC as identical
# at confidence >= 0.9. See vault/logs/2026-07-28.md.
#
# TOOL CALLING NEEDS BOTH FLAGS. Without them vLLM rejects any request carrying
# `tools` with a 400: '"auto" tool choice requires --enable-auto-tool-choice and
# --tool-call-parser to be set'. It is not a silent degradation to prose -- the
# call fails outright, so the planner cannot run at all. `hermes` is the parser
# for Qwen2.5's tool-call format.
#
# The model must already be in the HF cache; compute nodes have no outbound
# network. Pre-fetch on the LOGIN node:
#   python -c "from huggingface_hub import snapshot_download; \
#              snapshot_download('Qwen/Qwen2.5-72B-Instruct')"
# =============================================================================

set -euo pipefail

module load Python/3.9.6-GCCcore-11.2.0

MODEL="${LLM_MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
PORT="${LLM_PORT:-8000}"

echo "Node:  $(hostname)"
echo "Model: ${MODEL}"

# WHICH cards did Slurm give us, and are they healthy?
#
# compute-gpu-0-1 has one faulty A100: 00000000:CA:00.0 showed 27 uncorrectable
# ECC errors within 90 minutes of a reboot (805 aggregate) and, worse,
# remapped_rows.failure=1 -- its bad memory rows could not be retired. Job 48125
# died on it with "uncorrectable ECC error encountered" mid-request.
#
# TP=2 takes two of three cards, so whether a run touches the bad one depends on
# what else is scheduled. Logging bus ids and ECC counters up front means a
# failure is interpretable instead of mysterious, and a SUCCESS is too -- it may
# only mean we were allocated the two healthy cards.
echo "--- allocated GPUs ---"
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv
nvidia-smi --query-remapped-rows=gpu_bus_id,remapped_rows.uncorrectable,remapped_rows.failure --format=csv
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "----------------------"

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
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

#!/bin/bash
#SBATCH -p LIGHTGPU
#SBATCH --gres=gpu:1
#SBATCH --job-name=vllm-smoke
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_smoke_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# =============================================================================
# A SMALL tool-calling server, for verifying the planner's WIRE FORMAT only.
#
# Not for judging answers -- an 8B tells us nothing about what Qwen-72B will
# decide. What it does tell us, and what a scripted test cannot, is whether a
# real vLLM server emits tool calls our schemas accept, and whether our message
# shapes (assistant.tool_calls, role=tool, tool_call_id) round-trip correctly.
# Those break first and break silently.
#
# Runs on LIGHTGPU because compute-gpu-0-1 (the only node with real A100s) has
# been drained for a reboot since 2026-07-29. A single 20GB MIG slice holds an
# 8B at fp16 only just, hence the short context and high utilisation -- if it
# OOMs, drop --max-model-len to 4096 first.
#
# Tool calling needs BOTH flags. Without them vLLM ignores the `tools`
# parameter entirely and answers in prose, which looks like a model failure
# rather than a configuration one.
# =============================================================================

set -euo pipefail

module load Python/3.9.6-GCCcore-11.2.0

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
PORT="${LLM_PORT:-8001}"

echo "Node:  $(hostname)"
echo "Model: ${MODEL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

cd /home/xucabrjs/HEPCoverageKG
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"

export HF_HUB_OFFLINE=1

# Slurm hands out a MIG *UUID* in CUDA_VISIBLE_DEVICES, and vLLM's
# get_device_capability() resolves it as a plain device index --
# nvmlDeviceGetHandleByIndex then fails with NVMLError_InvalidArgument before
# the model is even loaded (job 48123). That call sits behind the V1-engine
# oracle check, so skipping V1 avoids it. If it fails again further down the
# stack, MIG is simply not usable for vLLM here and the only route is a real
# GPU on compute-gpu-0-1.
export VLLM_USE_V1=0

apptainer exec --nv \
    ~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" \
    --api-key "${LLM_API_KEY}" \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json

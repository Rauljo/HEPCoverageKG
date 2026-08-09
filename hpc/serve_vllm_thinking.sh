#!/bin/bash
# =============================================================================
# HEPCoverageKG: serve a REASONING model for the reference reader.
#
# Why a thinking model at all. The 24B reader is a literal reader -- measured
# three separate ways on 2026-08-09:
#   it will not invert "masses up to 875 GeV are excluded" into "lower limit"
#   it answered a corpus-wide question with "the text does not mention which..."
#   it matches surface topic over structural role (Higgs process vs Higgs object)
# Putting the reasoning field before the verdict in the JSON fixed 3 of 5 false
# positives for free. What it did NOT fix is recall: 2006.05880 says "A jet pair
# is tagged as a Higgs boson candidate if the neural network score..." and the
# reader still missed it. That is the failure a reasoning model is for.
#
# QwQ-32B-AWQ: Qwen's reasoning model, 19GB quantised, fits ONE A100-80GB. The
# 72B we already have cached is bigger but is NOT a reasoning model, so it tests
# a different variable -- run it as a separate arm, not as a substitute.
#
# --enable-reasoning + --reasoning-parser deepseek_r1 makes vLLM strip the
# <think> block server-side. The reader tolerates it either way, but stripping
# it here keeps the token accounting honest.
#
# 16k context, not 8k: the reasoning is emitted as ordinary output tokens, so a
# 12,000-char window plus a long chain of thought needs the headroom. Pair with
# READER_MAX_TOKENS=4000 on the client, or the reply truncates mid-thought and
# every call looks unparseable.
#
# Submit:
#   sbatch hpc/serve_vllm_thinking.sh
#   # then point the reader at it:
#   READER_MAX_TOKENS=4000 LLM_MODEL_NAME=Qwen/QwQ-32B-AWQ ...
# =============================================================================
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vllm-thinking
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_thinking_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"

MODEL="${THINKING_MODEL:-Qwen/QwQ-32B-AWQ}"
PORT="${VLLM_PORT:-8001}"          # 8001, so it can run BESIDE the 24B server
                                   # and both arms be compared without a restart
export HF_HUB_OFFLINE=1

echo "node  : $(hostname)"
echo "model : ${MODEL}"
echo "port  : ${PORT}"

# REFUSE A FAILING GPU BEFORE LOADING 19GB ONTO IT.
#
# compute-gpu-0-1 GPU 2 (GPU-bd028739-e189-8a3c-d439-85fadc134cc3) has 569
# volatile / 1413 aggregate uncorrected ECC errors. It loads a model and serves
# happily, then dies on the first real inference with
#     RuntimeError: CUDA error: uncorrectable ECC error encountered
# Two QwQ attempts died that way, at 7 and 4.5 minutes. Slurm still reports the
# node `mixed` with no drain reason, so it keeps handing the card out -- and
# with GPUs 0 and 1 busy, every single-GPU request lands on it.
#
# Third fault of this shape after D-040 (a slow A100) and D-049 (a node that
# accepts work and runs none): a resource that looks healthy until used. So test
# it, and exit non-zero in seconds so the allocation is released rather than
# burned. Submit with --requeue to cycle onto a different card.
nvidia-smi --query-gpu=index,uuid,name,ecc.errors.uncorrected.volatile.total \
           --format=csv,noheader
BAD=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total \
        --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1 ~ /^[0-9]+$/ && $1 > 0 {n++} END {print n+0}')
if [ "${BAD:-0}" -gt 0 ]; then
    echo "ERROR: this allocation includes a GPU with uncorrected ECC errors."
    echo "       Refusing to serve on it -- it would die on the first inference."
    echo "       Resubmit; with --requeue Slurm will try a different card."
    exit 1
fi
echo "GPU health: no uncorrected ECC errors"

apptainer exec --nv \
    ~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --api-key "${LLM_API_KEY}" \
    --served-model-name "${MODEL}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1

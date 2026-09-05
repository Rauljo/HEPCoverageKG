#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vllm-one
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_one_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
# =============================================================================
# ONE model on ONE card, serving both the planner and the critic.
#
# WHY IT EXISTS. serve_split.sh wants three cards so it can pick two clean ones,
# and the node has three A100s of which another user currently holds two. A job
# asking for three sits in "Resources" behind their whole array. A 32B at 4-bit
# is about 20GB and fits one card comfortably, so this takes what is actually
# free.
#
# THE COST, stated because it changes what a run means: with no second endpoint
# the critic falls back to the planner's own client, so planner and critic are
# the SAME model. Against the Qwen2.5 baseline that moves two variables at once
# -- planner 72B -> QwQ-32B and critic Llama-8B -> QwQ-32B -- and a difference
# cannot be attributed to either alone. It is a first look, not an ablation.
#
# THE ECC CHECK IS KEPT. This node has one A100 that accepts a model and dies on
# the first inference (D-040), and a single-card job has no spare to fall back
# to, so it refuses rather than discovering it mid-run.
# =============================================================================
set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
# THE SUBMITTED ENVIRONMENT MUST WIN OVER .env. `set -a; . ./.env` re-exports
# every key in that file, so `sbatch --export=ALL,LLM_MODEL_NAME=X` was silently
# overwritten by the model pinned in .env. Today that served the 72B to a client
# asking for QwQ -- one record, empty answer, and nothing saying which model had
# actually been loaded. Captured before the source, preferred after it.
_OVERRIDE_BIG="${LLM_MODEL_NAME:-}"
_OVERRIDE_SMALL="${CRITIC_MODEL:-}"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"
export HF_HUB_OFFLINE=1

MODEL="${_OVERRIDE_BIG:-${LLM_MODEL_NAME:-Qwen/QwQ-32B-AWQ}}"
IMG=~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif

echo "node: $(hostname)  model: $MODEL"
# A MIG-PARTITIONED NODE CANNOT HOST THESE MODELS, and says so here rather than
# through a confusing symptom. LIGHTGPU's compute-gpu-0-0 advertises gpu:a100:6,
# but those are six 3g.20gb slices of three cards. MIG instances cannot be
# pooled -- vLLM cannot tensor-parallel across them -- and QwQ-32B-AWQ is 19GB
# of weights against ~18GB usable, so it cannot load on one. Slurm hands out
# slice ids (12, 21) that `nvidia-smi -i` rejects, so the ECC probe returned
# "No devices were found" for every card and the job reported zero CLEAN cards.
# It read as failing hardware. It was the wrong partition.
if nvidia-smi -L 2>/dev/null | grep -q "MIG "; then
    echo "ERROR: $(hostname) is MIG-partitioned -- slices are too small."
    echo "       Submit to the GPU partition (whole cards), not LIGHTGPU."
    exit 76
fi

gpu="${CUDA_VISIBLE_DEVICES%%,*}"
[ -z "$gpu" ] && { echo "ERROR: no CUDA_VISIBLE_DEVICES"; exit 1; }
errs=$(nvidia-smi -i "$gpu" --query-gpu=ecc.errors.uncorrected.aggregate.total \
        --format=csv,noheader,nounits 2>/dev/null || echo unknown)
bus=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null || echo "?")
echo "  gpu $gpu ($bus): uncorrected ECC = $errs"
[ "$errs" = "0" ] || { echo "ERROR: card is not ECC-clean, refusing"; exit 75; }

# A REASONING MODEL NEEDS ROOM. QwQ emits its chain of thought before answering
# and those tokens come out of the same window; the reader once discarded 99 of
# 366 QwQ replies to a parser that did not expect them. 32k, and the completion
# allowance is raised on the client side.
apptainer exec --nv "$IMG" \
  vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --api-key "$LLM_API_KEY" \
  --served-model-name "$MODEL" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser hermes

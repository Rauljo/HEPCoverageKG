#!/bin/bash
# =============================================================================
# HEPCoverageKG: both readers, one allocation.
#
# WHY ONE JOB INSTEAD OF TWO.
#
# compute-gpu-0-1 has three A100s and one of them (bus 00000000:CA:00.0) is
# dead in the specific way that matters: it accepts a model, serves it, and
# throws `uncorrectable ECC error` on the first real inference. Slurm reports
# the node healthy and keeps handing the card out.
#
# So two separate one-GPU jobs cannot both get a good card -- there are only two
# good cards and any job holding a spare draws the bad one. Job 48365 did
# exactly that: submitted while a two-GPU QwQ job held the rest, it was handed
# CA:00.0 and refused to start, correctly and uselessly.
#
# Taking all three at once is the only allocation on this node where both models
# are guaranteed a healthy card. The bad one is held and never used, which costs
# nobody anything: it is unusable by any job until it is drained.
#
#   port 8000   Mistral-Small-24B   the literal reader
#   port 8001   QwQ-32B-AWQ         the reasoning reader, and the judge
#
# Both stay up for the whole reservation. The clients are ordinary CPU jobs on
# COMPUTE that talk to these two ports.
# =============================================================================
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:3
#SBATCH --job-name=vllm-both
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_both_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=24

set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"
export HF_HUB_OFFLINE=1

SMALL="${SMALL_MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
THINK="${THINKING_MODEL:-Qwen/QwQ-32B-AWQ}"
IMG=~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif

echo "node: $(hostname)"
nvidia-smi --query-gpu=index,pci.bus_id,ecc.errors.uncorrected.aggregate.total \
           --format=csv

# Choose only from within OUR allocation. nvidia-smi talks to NVML and ignores
# CUDA_VISIBLE_DEVICES, so a bare query lists cards other jobs own.
ALLOCATED="${CUDA_VISIBLE_DEVICES:-}"
[ -z "$ALLOCATED" ] && { echo "ERROR: no CUDA_VISIBLE_DEVICES -- refusing to guess"; exit 1; }
echo "allocated: ${ALLOCATED}"

# Aggregate, not volatile: volatile counters reset on driver reload, so a card
# that killed two jobs yesterday reads clean today on the volatile column.
HEALTHY=()
for gpu in ${ALLOCATED//,/ }; do
    errs=$(nvidia-smi -i "$gpu" \
             --query-gpu=ecc.errors.uncorrected.aggregate.total \
             --format=csv,noheader,nounits 2>/dev/null || echo unknown)
    bus=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null || echo "?")
    echo "  gpu ${gpu} (${bus}): uncorrected ECC = ${errs}"
    [ "$errs" = "0" ] && HEALTHY+=("$gpu")
done

if [ "${#HEALTHY[@]}" -lt 2 ]; then
    echo "ERROR: need two healthy cards, found ${#HEALTHY[@]}."
    exit 75   # EX_TEMPFAIL
fi
echo "serving on GPUs ${HEALTHY[0]} and ${HEALTHY[1]}"

serve () {   # gpu, model, port, extra args...
    local gpu="$1" model="$2" port="$3"; shift 3
    CUDA_VISIBLE_DEVICES="$gpu" apptainer exec --nv "$IMG" \
        vllm serve "$model" \
        --host 0.0.0.0 --port "$port" \
        --api-key "${LLM_API_KEY}" \
        --served-model-name "$model" \
        --gpu-memory-utilization 0.90 "$@" &
    echo "  launched ${model} on gpu ${gpu}, port ${port} (pid $!)"
}

# 8192 for the 24B: a 12,000-char window is ~3,470 tokens at the measured 3.46
# chars/token, so the prompt and a 500-token reply fit with room to spare.
serve "${HEALTHY[0]}" "$SMALL" 8000 --max-model-len 8192

# 16384 for QwQ: its reasoning is emitted as ordinary output tokens, so the same
# window plus a long chain of thought needs the headroom or every reply
# truncates mid-thought and reads as unparseable.
serve "${HEALTHY[1]}" "$THINK" 8001 --max-model-len 16384 \
      --enable-reasoning --reasoning-parser deepseek_r1

# If either server dies, the reservation is worthless -- fail loudly instead of
# leaving the survivor up and a client blocking on a port that will never open.
wait -n
echo "!!! one of the two servers exited; bringing the job down"
exit 1

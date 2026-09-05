#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=vllm-split
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_split_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
# =============================================================================
# The answerer and the judge on separate cards.
#
#   port 8000   Qwen2.5-72B-AWQ   TP=1   answers
#   port 8001   Llama-3.1-8B      TP=1   judges candidates
#
# WHY. The critic is 48% of all LLM calls, and its task is the narrow one: a
# short prompt, fifteen candidates, structured JSON back. Putting it on a small
# model frees the 72B to do the part that needs a large one.
#
# The 72B drops to TP=1 and so loses roughly half its KV cache -- but it also
# sheds roughly half its requests, so the two effects largely cancel, while the
# critic's half moves to a model several times faster per token with a card to
# itself. Net speed is an OPEN QUESTION and one of the two things this
# configuration exists to measure; the other is whether an 8B can judge at all.
#
# The caution on the second point is on the record: the 8B merged 8 distinct
# SMEFT Wilson coefficients on the aliases task and accepted WH/ZH as identical
# at confidence >= 0.9 (logs/2026-07-28.md). That is the neighbouring judgement,
# and it is why the GATE runs before any full arm.
#
# Both cards must be ECC-clean: this node has one A100 that accepts a model and
# dies on the first inference (D-040), and with only two healthy cards there is
# no spare to lose.
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

BIG="${_OVERRIDE_BIG:-${LLM_MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct-AWQ}}"
SMALL="${_OVERRIDE_SMALL:-${CRITIC_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}}"
IMG=~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif

echo "node: $(hostname)"
# Said out loud, because the silent version cost a run: the log showed the
# job starting and never which weights it loaded.
echo "serving BIG=${BIG}  SMALL=${SMALL}"
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

ALLOCATED="${CUDA_VISIBLE_DEVICES:-}"
[ -z "$ALLOCATED" ] && { echo "ERROR: no CUDA_VISIBLE_DEVICES -- refusing to guess"; exit 1; }

# Aggregate, not volatile: volatile counters reset on driver reload, so a card
# that killed two jobs yesterday reads clean today on the volatile column.
HEALTHY=()
for gpu in ${ALLOCATED//,/ }; do
    errs=$(nvidia-smi -i "$gpu" --query-gpu=ecc.errors.uncorrected.aggregate.total \
             --format=csv,noheader,nounits 2>/dev/null || echo unknown)
    bus=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null || echo "?")
    echo "  gpu ${gpu} (${bus}): uncorrected ECC = ${errs}"
    [ "$errs" = "0" ] && HEALTHY+=("$gpu")
done
[ "${#HEALTHY[@]}" -lt 2 ] && { echo "ERROR: need two clean cards, found ${#HEALTHY[@]}"; exit 75; }
echo "answerer on gpu ${HEALTHY[0]}, judge on gpu ${HEALTHY[1]}"

echo "--- clearing our own stale servers, if any ---"
pkill -u "$(id -u)" -f "vllm serve" 2>/dev/null && sleep 20 || echo "  none found"

CUDA_VISIBLE_DEVICES="${HEALTHY[0]}" apptainer exec --nv "$IMG" \
    vllm serve "$BIG" --host 0.0.0.0 --port 8000 --api-key "$LLM_API_KEY" \
    --served-model-name "$BIG" --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser hermes &
PID_BIG=$!
CUDA_VISIBLE_DEVICES="${HEALTHY[1]}" apptainer exec --nv "$IMG" \
    vllm serve "$SMALL" --host 0.0.0.0 --port 8001 --api-key "$LLM_API_KEY" \
    --served-model-name "$SMALL" --max-model-len 8192 --gpu-memory-utilization 0.90 &
PID_SMALL=$!

# NOT `wait -n` -- the bash on these nodes predates it, and under `set -e` that
# took a server job down 31 seconds after launch while both vLLM processes
# carried on as orphans, Slurm reporting the node idle with its cards free.
while kill -0 "$PID_BIG" 2>/dev/null && kill -0 "$PID_SMALL" 2>/dev/null; do
    sleep 30
done
echo "!!! one of the two servers exited; bringing the job down"
kill "$PID_BIG" "$PID_SMALL" 2>/dev/null || true
exit 1

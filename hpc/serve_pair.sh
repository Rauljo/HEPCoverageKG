#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=vllm-pair
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_pair_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
# =============================================================================
# BOTH models on ONE card. The planner on 8000, the judge on 8001.
#
# WHY. serve_split.sh and serve_both.sh each take all three A100s, for a reason
# that was correct when written: one card (bus CA:00.0) throws uncorrectable ECC
# on first inference, so a job wanting two GOOD cards must hold all three or
# risk drawing the dead one. But that only works when the node is ours. Another
# user's array now holds two cards with a 3-day limit, leaving exactly one free,
# and a three-card request queues until 2026-09-03.
#
# The arithmetic says one card is enough and always was:
#
#   A100-80GB                      81,920 MiB
#   QwQ-32B-AWQ  weights ~19GB     0.45 util -> ~36GB   (~17GB for KV at 32k)
#   Llama-3.1-8B weights ~16GB     0.32 util -> ~26GB   (~10GB for KV at 8k)
#                                  0.77 total, headroom left deliberately
#
# WHAT IT COSTS. Two vLLM processes share the card's compute, so both are slower
# than they would be alone -- they interleave rather than run in parallel. That
# is a throughput cost, not a correctness one, and it buys the thing that
# matters: planner and judge stay DIFFERENT models. serve_one.sh's fallback
# makes them the same model, which confounds every critic comparison.
#
# THE ECC CHECK STILL DECIDES. With one card there is no spare, so if the free
# card is the dead one this refuses in a second rather than dying on the first
# inference an hour in.
# =============================================================================
set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
_OVERRIDE_BIG="${LLM_MODEL_NAME:-}"
_OVERRIDE_SMALL="${CRITIC_MODEL:-}"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"
export HF_HUB_OFFLINE=1

BIG="${_OVERRIDE_BIG:-${LLM_MODEL_NAME:-Qwen/QwQ-32B-AWQ}}"
SMALL="${_OVERRIDE_SMALL:-${CRITIC_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}}"
IMG="${VLLM_IMAGE:-$HOME/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif}"
BIG_UTIL="${BIG_UTIL:-0.45}"
SMALL_UTIL="${SMALL_UTIL:-0.32}"
# PORTS AND THE KILL ARE PARAMETERS BECAUSE A SECOND SERVER MAY SHARE THE NODE.
# compute-gpu-0-1 has three cards; one job can hold two and leave one usable.
# Started blind, this script would `pkill vllm serve` -- killing the OTHER job's
# model -- and then fail to bind 8000 because that job already has it.
BIG_PORT="${BIG_PORT:-8000}"
SMALL_PORT="${SMALL_PORT:-8001}"

echo "node: $(hostname)"
echo "serving BIG=${BIG} on :${BIG_PORT} (util ${BIG_UTIL})  SMALL=${SMALL} on :${SMALL_PORT} (util ${SMALL_UTIL})"

if nvidia-smi -L 2>/dev/null | grep -q "MIG "; then
    echo "ERROR: $(hostname) is MIG-partitioned -- slices are too small."; exit 76
fi
# ASK FOR TWO, USE THE FIRST CLEAN ONE. Both models fit on one card, so the
# second is requested purely as insurance against being handed CA:00.0 -- which
# is exactly what happened to the one-card version of this job: the only free
# card on the node was the dead one and it refused, correctly and uselessly.
# The spare is held and never used, which costs nobody anything, because the bad
# card is unusable by any job until it is drained.
ALLOCATED="${CUDA_VISIBLE_DEVICES:-}"
[ -z "$ALLOCATED" ] && { echo "ERROR: no CUDA_VISIBLE_DEVICES"; exit 1; }
CARD=""
for g in ${ALLOCATED//,/ }; do
    errs=$(nvidia-smi -i "$g" --query-gpu=ecc.errors.uncorrected.aggregate.total \
            --format=csv,noheader,nounits 2>/dev/null || echo unknown)
    bus=$(nvidia-smi -i "$g" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null || echo "?")
    echo "  gpu $g ($bus): uncorrected ECC = $errs"
    [ "$errs" = "0" ] && [ -z "$CARD" ] && CARD="$g"
done
[ -n "$CARD" ] || { echo "ERROR: no ECC-clean card in this allocation, refusing"; exit 75; }

# TWO CLEAN CARDS -> one each. ONE CLEAN CARD -> share it, if the pair fits.
# Sharing was the point of this script when both models were small (QwQ 19GB +
# Llama-8B 16GB on 80GB). An unquantised 27B is 55.6GB and does NOT leave room
# for the judge, so with only one clean card the big model runs alone and the
# job says so rather than OOMing halfway through a run.
CLEAN=()
for g in ${ALLOCATED//,/ }; do
    e=$(nvidia-smi -i "$g" --query-gpu=ecc.errors.uncorrected.aggregate.total \
         --format=csv,noheader,nounits 2>/dev/null || echo unknown)
    [ "$e" = "0" ] && CLEAN+=("$g")
done
BIG_CARD="${CLEAN[0]}"
if [ "${#CLEAN[@]}" -ge 2 ]; then
    SMALL_CARD="${CLEAN[1]}"
    BIG_UTIL="${BIG_UTIL:-0.90}"
    SMALL_UTIL="${SMALL_UTIL:-0.85}"
    echo "two clean cards: BIG on gpu ${BIG_CARD}, SMALL on gpu ${SMALL_CARD}"
else
    SMALL_CARD=""
    echo "only one clean card (gpu ${BIG_CARD}) -- serving BIG alone; the judge"
    echo "endpoint on 8001 will NOT come up, so run critic-off arms only."
fi

if [ "${KILL_STALE:-1}" = "1" ]; then
    echo "--- clearing our own stale servers, if any ---"
    pkill -u "$(id -u)" -f "vllm serve" 2>/dev/null && sleep 20 || echo "  none found"
else
    echo "--- KILL_STALE=0: leaving other servers alone (sharing the node) ---"
fi

CUDA_VISIBLE_DEVICES="$BIG_CARD" apptainer exec --nv "$IMG" vllm serve "$BIG" \
    --host 0.0.0.0 --port "$BIG_PORT" --api-key "$LLM_API_KEY" \
    --served-model-name "$BIG" --max-model-len "${BIG_LEN:-32768}" \
    --gpu-memory-utilization "$BIG_UTIL" \
    --enable-auto-tool-choice --tool-call-parser hermes &
PID_BIG=$!

PID_SMALL=""
if [ -n "$SMALL_CARD" ]; then
    sleep 90      # let the big model claim its memory first
    CUDA_VISIBLE_DEVICES="$SMALL_CARD" apptainer exec --nv "$IMG" vllm serve "$SMALL" \
        --host 0.0.0.0 --port "$SMALL_PORT" --api-key "$LLM_API_KEY" \
        --served-model-name "$SMALL" --max-model-len 8192 \
        --gpu-memory-utilization "$SMALL_UTIL" &
    PID_SMALL=$!
fi

while kill -0 "$PID_BIG" 2>/dev/null \
      && { [ -z "$PID_SMALL" ] || kill -0 "$PID_SMALL" 2>/dev/null; }; do
    sleep 30
done
echo "!!! a server exited; bringing the job down"
kill "$PID_BIG" ${PID_SMALL:+$PID_SMALL} 2>/dev/null || true
exit 1

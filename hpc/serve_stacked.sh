#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vllm-stack
#SBATCH --requeue
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_stack_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
# =============================================================================
# THE ANSWERER AND THE JUDGE ON ONE CARD.
#
#   port 8000   the answerer   (QwQ-32B-AWQ by default)
#   port 8001   the judge      (Qwen3.5-9B by default)
#
# WHY THIS EXISTS. Every other pair script wants two GPUs, and on 2026-09-06
# there was exactly one to be had. compute-gpu-0-1 has three A100s: another
# user held one, we held one, and the third is the card with 1413 uncorrected
# ECC errors. The judge requeued onto that same broken card four times, gave
# up, and the four arms behind it timed out waiting for an endpoint that was
# never going to appear.
#
# The arithmetic says one card is enough and always did. From the answerer's
# own log on that day:
#
#     Model loading took 18.1494 GiB          <- QwQ-32B-AWQ weights
#     GPU KV cache size: 212,528 tokens       <- at --gpu-memory-utilization 0.90
#
# 18 GiB of weights on an 80 GiB card, and a quarter of a million tokens of KV
# cache for a run that never has more than eight requests in flight. The 0.90
# was never sized from the workload; it was the default. Two models at 0.45 and
# 0.35 leave both with far more cache than they use and 16 GiB spare.
#
# WHAT THIS COSTS. The two models share memory bandwidth and SMs, so both are
# slower than they would be alone -- somewhere around 20-30% on a mixed load,
# unmeasured here. That is the price of running at all, and it is a better
# trade than an idle GPU: the previous configuration spent 8h29m queueing and
# produced nothing.
#
# WHAT IT DOES NOT CHANGE. Same weights, same ports, same api key, same
# `enable_thinking=false` requirement on the judge (D-105) -- that is a
# per-request field and lives in the client, not here. A run cannot tell this
# apart from the two-card version except in wall time, which is why it is safe
# to compare arms across the two. It is NOT safe to compare TIMINGS across
# them, and `seconds` in a run record is exactly such a timing.
# =============================================================================
set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"

# The submitted environment must win over .env (D-082).
_OVERRIDE_BIG="${LLM_MODEL_NAME:-}"
_OVERRIDE_SMALL="${CRITIC_MODEL:-}"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"
export HF_HUB_OFFLINE=1

BIG="${_OVERRIDE_BIG:-${LLM_MODEL_NAME:-Qwen/QwQ-32B-AWQ}}"
# THE .env PINS THE 72B, AND THE DEFAULT ABOVE NEVER GETS A CHANCE.
# 2026-09-12: submitted with no --export, this loaded Qwen2.5-72B-Instruct-AWQ
# because `set -a; . ./.env` had already exported LLM_MODEL_NAME. Stacked with
# the 9B it does not fit -- "Available KV cache memory: -7.07 GiB" -- and the
# job died six minutes in, after the eval arms had already started against it.
# The default here is the intended model, so say so rather than fail obscurely.
case "$BIG" in
  *72B*)
    echo "WARNING: serving $BIG. Stacked with $SMALL this has historically"
    echo "         failed with negative KV cache. Pass LLM_MODEL_NAME on the"
    echo "         sbatch line if you meant the 32B." ;;
esac
SMALL="${_OVERRIDE_SMALL:-${CRITIC_MODEL:-Qwen/Qwen3.5-9B}}"
IMG="${VLLM_IMAGE:-$HOME/hepcoveragekg_setup/images/vllm-openai-v0.18.0}"
BIG_UTIL="${BIG_UTIL:-0.45}"
SMALL_UTIL="${SMALL_UTIL:-0.35}"
BIG_LEN="${BIG_LEN:-32768}"
SMALL_LEN="${SMALL_LEN:-8192}"
# Qwen3.5 emits its chain of thought before the answer; unparsed, those tokens
# land in the reply body. Empty for a model that does not think.
PARSER="${CRITIC_REASONING_PARSER-qwen3}"

echo "node: $(hostname)"
echo "serving BIG=${BIG} (:8000, util ${BIG_UTIL})  SMALL=${SMALL} (:8001, util ${SMALL_UTIL})"

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
[ "$errs" = "0" ] || {
    echo "ERROR: card is not ECC-clean ($errs uncorrected)."
    tries="${SLURM_RESTART_COUNT:-0}"
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "$tries" -lt "${MAX_ECC_REQUEUE:-4}" ]; then
        echo "       requeueing (attempt $((tries + 1)) of ${MAX_ECC_REQUEUE:-4})"
        scontrol requeue "$SLURM_JOB_ID"
        sleep 30
        exit 0
    fi
    echo "       giving up after $tries requeues: every card drawn was faulty."
    exit 75
}

echo "--- clearing our own stale servers, if any ---"
pkill -u "$(id -u)" -f "vllm serve" 2>/dev/null && sleep 20 || echo "  none found"

# BIG FIRST, AND WAIT FOR IT. vLLM sizes its KV cache from free memory at
# startup, so two servers racing to profile the same card both see it empty and
# both claim their share of a total that no longer exists. Starting the second
# only once the first is answering makes the split deterministic.
apptainer exec --nv --pwd /tmp \
  --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 "$IMG" \
  vllm serve "$BIG" --host 0.0.0.0 --port 8000 --api-key "$LLM_API_KEY" \
    --served-model-name "$BIG" --max-model-len "$BIG_LEN" \
    --gpu-memory-utilization "$BIG_UTIL" \
    --enable-auto-tool-choice --tool-call-parser hermes &
PID_BIG=$!

echo "--- waiting for the answerer to finish loading before starting the judge ---"
ready=0
for _ in $(seq 1 120); do          # 20 minutes
    if printf 'header = "Authorization: Bearer %s"\nurl = "http://127.0.0.1:8000/v1/models"\n' \
         "$LLM_API_KEY" | curl -sf --max-time 10 -o /dev/null -K -; then
        echo "  answerer ready"; ready=1; break
    fi
    kill -0 "$PID_BIG" 2>/dev/null || { echo "FATAL: answerer died while loading"; exit 1; }
    sleep 10
done
[ "$ready" = 1 ] || { echo "FATAL: answerer never answered"; kill "$PID_BIG" 2>/dev/null; exit 1; }

SMALL_ARGS=(--host 0.0.0.0 --port 8001 --api-key "$LLM_API_KEY"
            --served-model-name "$SMALL" --max-model-len "$SMALL_LEN"
            --gpu-memory-utilization "$SMALL_UTIL")
[ -n "$PARSER" ] && SMALL_ARGS+=(--reasoning-parser "$PARSER")
apptainer exec --nv --pwd /tmp \
  --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 "$IMG" \
  vllm serve "$SMALL" "${SMALL_ARGS[@]}" &
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

#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vllm-rewriter
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_rewriter_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# =============================================================================
# Serve the REWRITER model (S-67), not the answerer.
#
# Deliberately different from serve_vllm_70b.sh in three ways:
#
#   no tool-calling flags.  Rewriting is one prompt in, one sentence out. The
#       `--tool-call-parser hermes` in the 70B script is Qwen-specific and would
#       be wrong here anyway; Mistral uses a different parser.
#
#   8k context, not 32k.  A rewrite prompt is a few hundred tokens, so a long
#       per-request cap would only shrink concurrency for nothing.
#
#   4h wall, 128G.  This is a short batch job, not a day-long server.
#
# WHY a different model family at all: the rewriter must not share vocabulary
# with the answerer (S-37/S-67). Extraction is Claude Sonnet, answering is
# Qwen-72B, so Mistral is a genuine third lineage -- and questions phrased by the
# same model that answers them would be phrased in whatever wording that model
# already finds easiest.
# =============================================================================

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
# Why AWQ rather than fp16: 72B fp16 is ~145GB of weights and would need at
# least two cards. AWQ is ~39GB, which fits on ONE 80GB A100 with ~35GB left for
# KV cache.
#
# Why 32k context (changed 2026-07-31): 8192 was too small. ONE question already
# cost 11,902 prompt tokens across three rounds, because every round re-sends the
# whole conversation, and a six-round question or a follow-up would overflow. The
# cost is concurrency, not memory -- the KV cache is fixed by
# gpu-memory-utilization, so a longer per-request cap simply means fewer requests
# in flight: ~97k cache / 32k = ~3 concurrent instead of ~12. Fine for one person
# debugging; revisit before batch evaluation, where TP=2 (~320k cache) buys the
# concurrency back.
#
# Why TP=1 rather than TP=2 (changed 2026-07-31): one of the three A100s is
# faulty, so TP=2 takes two of three cards and is very likely to include it --
# it did, twice. TP=1 draws one card, so a healthy allocation is the common case
# rather than the lucky one, and the check below turns a bad draw into a
# ten-second exit instead of a five-minute crash. Revert to TP=2 for throughput
# once the bad card is out of service.
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

MODEL="${REWRITER_MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
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

# Refuse a known-bad card immediately.
#
# 00000000:CA:00.0 on compute-gpu-0-1 fails with "uncorrectable ECC error
# encountered" on the FIRST inference request -- reproduced twice (48125, 48128),
# both times ~5 minutes in. Without this check a bad draw costs a full model load
# and then dies mid-request; with it, the job exits in seconds and can be
# resubmitted onto a different card.
#
# Checked by BUS ID, not by index: Slurm renumbers visible devices per job, so
# "GPU 2" means nothing inside the allocation. The bus id is stable.
#
# And checked ONLY for the cards we were actually given. `nvidia-smi` talks to
# NVML directly and IGNORES CUDA_VISIBLE_DEVICES, so a bare query lists every
# card on the node -- an earlier version of this check did exactly that and
# refused a perfectly healthy allocation (job 48129 held 65:00.0 and was turned
# away because CA:00.0 exists elsewhere in the box). `-i` restricts it.
BAD_GPUS="00000000:CA:00.0"
VISIBLE="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "${VISIBLE}" ]; then
    echo "CUDA_VISIBLE_DEVICES is unset; cannot tell which card we hold -- proceeding"
    ALLOCATED=""
else
    ALLOCATED=$(nvidia-smi -i "${VISIBLE}" --query-gpu=pci.bus_id --format=csv,noheader)
fi
echo "allocated bus ids: ${ALLOCATED:-<unknown>}"
for bus in ${ALLOCATED}; do
    for bad in ${BAD_GPUS}; do
        if [ "${bus}" = "${bad}" ]; then
            echo "REFUSING TO START: allocated known-faulty GPU ${bus}"
            echo "  (uncorrectable ECC, remapped_rows.failure=1 -- see vault D-040)"
            echo "  resubmit; with 3 cards there is a good chance of a healthy draw."
            exit 75   # EX_TEMPFAIL: transient, worth retrying
        fi
    done
done
# The bus-id blocklist above only knows the card D-040 named. A SECOND card has
# since gone bad the same way -- compute-gpu-0-1 GPU 2, 1413 aggregate
# uncorrected ECC errors -- and Slurm reports the node healthy, so it keeps
# handing it out. A blocklist needs updating after each failure; the error
# counter is the actual symptom, so read that instead and let the list stay as
# the belt to this braces.
#
# Volatile counters reset on driver reload, aggregate ones do not. Only the
# aggregate can be trusted for "has this card ever failed".
for gpu in ${VISIBLE//,/ }; do
    errs=$(nvidia-smi -i "$gpu" \
             --query-gpu=ecc.errors.uncorrected.aggregate.total \
             --format=csv,noheader,nounits 2>/dev/null || echo unknown)
    echo "  gpu ${gpu}: uncorrected ECC (aggregate) = ${errs}"
    if [ "$errs" != "0" ] && [ "$errs" != "unknown" ] && [ "$errs" != "N/A" ]; then
        echo "REFUSING TO START: gpu ${gpu} has ${errs} uncorrected ECC errors."
        echo "  It loads a model happily and dies on the first real inference."
        exit 75   # EX_TEMPFAIL: transient, worth retrying onto another card
    fi
done
echo "GPU health check passed"
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
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "${LLM_API_KEY}" \
    --served-model-name "${MODEL}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
